"""
Dedalus script for 2D Rayleigh-Benard convection with configurable parameters.
"""

import numpy as np
from mpi4py import MPI
from scipy.special import erf
import time
import pathlib
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
import logging
logger = logging.getLogger(__name__)


def run_rayleigh_benard_simulation(
    # Domain parameters
    Lx=3.0, Lz=1.0,
    # Physical parameters
    Pr=1.0, Ra=1e7,
    # Time-periodic boundary condition parameters
    omega=1.0, A=1.0,
    # Numerical parameters
    nx=600, nz=280,
    # Time integration parameters
    initial_dt=1e-10, max_dt=3e-3,
    stop_sim_time=None, stop_wall_time=np.inf, stop_iteration=np.inf,
    # CFL parameters
    cfl_safety=0.3, cfl_max_change=1.5, cfl_min_change=0.1, cfl_threshold=0.1,
    # Output parameters
    output_dir='.',
    snapshot_dt=None, analysis_dt=None, profile_dt=None,
    max_snapshot_writes=100, max_analysis_writes=500, max_profile_writes=500,
    # Initial condition parameters
    perturbation_amplitude=1e-3, random_seed=42,
    # Logging parameters
    log_cadence=20
):
    """
    Run a 2D Rayleigh-Bénard convection simulation with time-periodic heating.
    
    Parameters:
    -----------
    Domain parameters:
        Lx, Lz : float
            Domain size in x and z directions
    
    Physical parameters:
        Pr : float
            Prandtl number (momentum diffusivity / thermal diffusivity)
        Ra : float
            Rayleigh number (buoyancy forces / viscous forces)
    
    Time-periodic boundary condition parameters:
        omega : float
            Frequency of temperature oscillation at bottom boundary
        A : float
            Amplitude of temperature oscillation at bottom boundary
            Bottom BC: T = 1 + A*sin(2*π*omega*t)
    
    Numerical parameters:
        nx, nz : int
            Number of grid points in x and z directions
    
    Time integration parameters:
        initial_dt : float
            Initial timestep
        max_dt : float
            Maximum allowed timestep
        stop_sim_time : float or None
            Simulation time to stop at (if None, uses 500*τ where τ=1/ω)
        stop_wall_time : float
            Wall clock time to stop at
        stop_iteration : int
            Number of iterations to stop at
    
    CFL parameters:
        cfl_safety : float
            CFL safety factor
        cfl_max_change : float
            Maximum allowed change in timestep
        cfl_min_change : float
            Minimum allowed change in timestep
        cfl_threshold : float
            CFL threshold
    
    Output parameters:
        output_dir : str
            Directory for output files
        snapshot_dt, analysis_dt, profile_dt : float or None
            Output intervals (if None, uses default fractions of τ)
        max_*_writes : int
            Maximum number of output files for each type
    
    Initial condition parameters:
        perturbation_amplitude : float
            Amplitude of random perturbations
        random_seed : int
            Seed for random number generator
    
    Logging parameters:
        log_cadence : int
            How often to log iteration info
    
    Returns:
    --------
    solver : dedalus solver object
        The solved system (for post-processing if needed)
    """
    
    # Derived parameters
    tau = 1.0 / omega
    
    # Set default output intervals if not specified
    if stop_sim_time is None:
        stop_sim_time = 500.0 * tau
    if snapshot_dt is None:
        snapshot_dt = 0.1 * tau
    if analysis_dt is None:
        analysis_dt = 0.01 * tau
    if profile_dt is None:
        profile_dt = 0.01 * tau
    
    logger.info(f"Starting Rayleigh-Bénard simulation with Ra={Ra}, Pr={Pr}")
    logger.info(f"Domain: {Lx} x {Lz}, Grid: {nx} x {nz}")
    logger.info(f"Time-periodic BC: A={A}, ω={omega}, τ={tau}")
    
    # Create bases and domain
    x_basis = de.Fourier('x', nx, interval=(0, Lx))
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz))
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

    x = domain.grid(0)
    z = domain.grid(1)

    # 2D Boussinesq hydrodynamics
    problem = de.IVP(domain, variables=['p', 'T', 'u', 'w', 'uz', 'wz', 'Tz'])
    problem.meta[:]['z']['dirichlet'] = True

    # Add all parameters
    problem.parameters['A'] = A
    problem.parameters['ω'] = omega
    problem.parameters['τ'] = tau
    problem.parameters['Lx'] = Lx
    problem.parameters['Lz'] = Lz
    problem.parameters['Pr'] = Pr
    problem.parameters['Ra'] = Ra
    problem.parameters['π'] = np.pi
    problem.parameters['c1'] = np.sqrt(Pr/Ra)
    problem.parameters['c2'] = 1./np.sqrt(Pr*Ra)
    problem.parameters['c3'] = np.sqrt(Pr*Ra)
    problem.parameters['c4'] = np.sqrt(Ra/Pr)

    # Add equations
    problem.add_equation("dx(u) + wz = 0")
    problem.add_equation("dt(u) - c1*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
    problem.add_equation("dt(w) - c1*(dx(dx(w)) + dz(wz)) + dz(p) - T = -(u*dx(w) + w*wz)")
    problem.add_equation("dt(T) - c2*(dx(dx(T)) + dz(Tz)) = -(u*dx(T) + w*Tz)")
    problem.add_equation("Tz - dz(T) = 0")
    problem.add_equation("uz - dz(u) = 0")
    problem.add_equation("wz - dz(w) = 0")

    # Boundary conditions
    problem.add_bc("left(T) = 1 + A*sin(2*π*ω*t)")
    problem.add_bc("left(u) = 0")
    problem.add_bc("left(w) = 0")
    problem.add_bc("right(T) = 0")
    problem.add_bc("right(u) = 0")
    problem.add_bc("right(w) = 0", condition="(nx != 0)")
    problem.add_bc("integ(p) = 0", condition="(nx == 0)")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    logger.info('Solver built')

    # Initial conditions
    T = solver.state['T']
    Tz = solver.state['Tz']

    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=random_seed)
    noise = rand.standard_normal(gshape)[slices]

    # Perturbations damped at walls
    zb, zt = z_basis.interval
    pert = perturbation_amplitude * noise * (zt - z) * (z - zb)
    T['g'] = pert + 1 - z 
    T.differentiate('z', out=Tz)

    # Set integration parameters
    solver.stop_sim_time = stop_sim_time
    solver.stop_wall_time = stop_wall_time
    solver.stop_iteration = stop_iteration

    # Analysis outputs
    snaps = solver.evaluator.add_file_handler(
        f'{output_dir}/snapshots', sim_dt=snapshot_dt, max_writes=max_snapshot_writes
    )
    snaps.add_task("u", name='u')
    snaps.add_task("w", name='w')
    snaps.add_task("T", name='T')

    analysis = solver.evaluator.add_file_handler(
        f'{output_dir}/analysis', sim_dt=analysis_dt, max_writes=max_analysis_writes
    )
    analysis.add_task("(c3*integ(integ(w*T, 'x'), 'z') - integ(integ(Tz, 'x'), 'z'))/(Lx*Lz)", name='Nu_global')
    analysis.add_task("c4*sqrt(integ(integ(u*u+w*w, 'x'), 'z')/(Lx*Lz))", name='Re_global')
    analysis.add_task("sqrt(integ(integ(u*u, 'x'), 'z')/(Lx*Lz))", name='urms_global')
    analysis.add_task("sqrt(integ(integ(w*w, 'x'), 'z')/(Lx*Lz))", name='wrms_global')
    analysis.add_task("integ(integ(w*T, 'x'), 'z')/(Lx*Lz)", name='wT_global')
    analysis.add_task("integ(integ(Tz, 'x'), 'z')/(Lx*Lz)", name='Tz_global')

    profiles = solver.evaluator.add_file_handler(
        f'{output_dir}/profile', sim_dt=profile_dt, max_writes=max_profile_writes
    )
    profiles.add_task("integ(w*T, 'x')/Lx", name='wT_profile')
    profiles.add_task("integ(Tz, 'x')/Lx", name='Tz_profile')
    profiles.add_task("integ(T, 'x')/Lx", name='T_profile')
    profiles.add_task("sqrt(integ(u*u, 'x')/Lx)", name='urms_profile')
    profiles.add_task("sqrt(integ(w*w, 'x')/Lx)", name='wrms_profile')
    profiles.add_task("interp(T, z=0.5)", name='centralT')

    # CFL
    CFL = flow_tools.CFL(
        solver, initial_dt=initial_dt, cadence=log_cadence, 
        safety=cfl_safety, max_change=cfl_max_change, min_change=cfl_min_change, 
        max_dt=max_dt, threshold=cfl_threshold
    )
    CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=log_cadence)
    flow.add_property("sqrt(u*u + w*w)", name='Re')

    # Main loop
    try:
        logger.info('Starting main integration loop')
        start_time = time.time()
        while solver.ok:
            dt = CFL.compute_dt()
            solver.step(dt)
            if (solver.iteration-1) % log_cadence == 0:
                logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
                logger.info('Max Re = %f' %flow.max('Re'))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i' %solver.iteration)
        logger.info('Sim end time: %f' %solver.sim_time)
        logger.info('Run time: %.2f sec' %(end_time-start_time))
        logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

    return solver


# Example usage:
if __name__ == "__main__":
    # Run with default parameters
    solver = run_rayleigh_benard_simulation()
    
    # Or customize parameters:
    # solver = run_rayleigh_benard_simulation(
    #     Ra=1e6,                    # Lower Rayleigh number
    #     Pr=0.7,                    # Air-like Prandtl number
    #     omega=2.0,                 # Higher frequency oscillation
    #     A=0.5,                     # Smaller amplitude
    #     nx=400, nz=200,            # Lower resolution
    #     stop_sim_time=100.0,       # Shorter simulation
    #     output_dir='./custom_run'  # Custom output directory
    # )
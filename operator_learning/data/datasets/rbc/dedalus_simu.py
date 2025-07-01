import os
import numpy as np
from datetime import datetime
from time import sleep
from mpi4py import MPI
import dedalus.public as d3


COMM_WORLD = MPI.COMM_WORLD
MPI_SIZE = COMM_WORLD.Get_size()
MPI_RANK = COMM_WORLD.Get_rank()

def runSimu(dirName, nDim=2, Rayleigh=1e7, resFactor=1, baseDt=1e-2/2, seed=999,
           tBeg=0, tEnd=150, dtWrite=0.1, writeVort=False, writeFull=False,
           initFields=None, writeFields=True, writeSpaceDistr=False, 
           logEvery=100, distrMesh=None):

    """
    Run Rayleigh–Bénard convection (RBC) simulation 2-D or 3-D in a given folder.

    Args:
    dirName (str): Name of directory where snapshots and run info will be stored.
    nDim (int): 2-D or 3-D
    Rayleigh (float): Rayleigh number.
    resFactor (int): Resolution factor, based on a base grid size of (256, 64).
    baseDt (float): Base time-step for the base space resolution. Defaults to 1e-2/2.
    initFields (dict, optional): Dictionary containing initial conditions. Defaults to None.
    seed (int, optional): Seed for the random noise in the initial solution. Defaults to 999.
    tBeg (float): Simulation start time. Defaults to 0.
    tEnd (float): Simulation end time.
    dtWrite (float): Time interval between saved snapshots.
    writeFields (bool): Save snapshots. Defaults to True.
    writeVort (bool, optional): If True, write vorticity to snapshots. Defaults to False.
    writeFull (bool, optional): If True, write Tau variables to snapshots. Defaults to False.
    logEvery (int): Frequency of logging output to the console. Defaults to 100 iterations.
    writeSpaceDistr (bool, optional): If True, write spatial parallel distribution file (for Dedalus). Defaults to False.
    distrMesh (optional): Dedalus MPI mesh specification. Defaults to None.
    
    """

    assert nDim in [2, 3], "nDim must be 2 or 3"

    if nDim == 2:
        Lx, Lz = 4, 1
        Nx, Nz = 256 * resFactor, 64 * resFactor
    else:
        Lx = Ly = Lz = 1
        Nx = Ny = Nz = 64*resFactor
    timestep = baseDt / resFactor
    nSteps = round((tEnd - tBeg) / timestep, ndigits=3)
    if round(nSteps * timestep, ndigits=3) != (tEnd - tBeg):
        raise ValueError(f"tEnd ({tEnd}) is not divisible by timestep ({timestep})")

    nSteps = int(nSteps)
    infos = {"nSteps": nSteps + 1,
             "nDOF": Nx * Nz
            }

    Prandtl = 1.0
    dealias = 3 / 2
    stop_sim_time = tEnd
    timestepper = d3.RK443
    dtype = np.float64
    sComm = COMM_WORLD

    if os.path.isfile(f"{dirName}/01_finalized.txt"):
        if MPI_RANK == 0:
            print(" -- simulation already finalized, skipping !")
        return infos, None

    def log(msg):
        if MPI_RANK == 0:
            with open(f"{dirName}/simu.log", "a") as f:
                f.write(f"{dirName} -- ")
                f.write(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
                f.write(f", MPI rank {MPI_RANK} ({MPI_SIZE}) : {msg}\n")

    os.makedirs(dirName, exist_ok=True)
    with open(f"{dirName}/00_infoSimu.txt", "w") as f:
        f.write(f"Rayleigh : {Rayleigh:1.2e}\n")
        f.write(f"Seed : {seed}\n")
        f.write(f"Nx, Nz : {Nx}, {Nz}\n")
        f.write(f"dt : {timestep:1.2e}\n")
        f.write(f"tEnd : {stop_sim_time}\n")
        f.write(f"dtWrite : {dtWrite}\n")

    if nDim == 2:
        coords = d3.CartesianCoordinates('x', 'z')
        distr = d3.Distributor(coords, dtype=dtype, mesh=distrMesh, comm=sComm)
        xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
        zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
        bases = (xbasis, zbasis)
        x, z = distr.local_grids(xbasis, zbasis)
        ex, ez = coords.unit_vector_fields(distr)
    else:
        coords = d3.CartesianCoordinates('x', 'y', 'z')
        distr = d3.Distributor(coords, dtype=dtype, mesh=distrMesh, comm=sComm)
        xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
        ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
        zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)
        bases = (xbasis, ybasis, zbasis)
        x, y, z = distr.local_grids(xbasis, ybasis, zbasis)
        ex, ey, ez = coords.unit_vector_fields(distr)

    tau_bases = bases[:-1]

    # Fields
    p = distr.Field(name='p', bases=bases)
    b = distr.Field(name='b', bases=bases)
    u = distr.VectorField(coords, name='u', bases=bases)

    tau_p = distr.Field(name='tau_p')
    tau_b1 = distr.Field(name='tau_b1', bases=tau_bases)
    tau_b2 = distr.Field(name='tau_b2', bases=tau_bases)
    tau_u1 = distr.VectorField(coords, name='tau_u1', bases=tau_bases) 
    tau_u2 = distr.VectorField(coords, name='tau_u2', bases=tau_bases)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-0.5)
    nu = (Rayleigh / Prandtl)**(-0.5)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez * lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez * lift(tau_b1) # First-order reduction

    if writeSpaceDistr:
        MPI.COMM_WORLD.Barrier()
        sleep(0.01 * MPI_RANK)  # Small delay to reduce interleaved prints
        print(f"Rank {MPI_RANK}({MPI_SIZE}) :\n"
              f"\tx: {x.shape}, [{x.min(initial=np.inf)}, {x.max(initial=-np.inf)}]\n"
              f"\tz: {z.shape}, [{z.min(initial=np.inf)}, {z.max(initial=-np.inf)}]\n", flush=True)
        if nDim == 3:
            print(f"\t y: shape={y.shape}, range=[{y.min(initial=np.inf)}, {y.max(initial=-np.inf)}]", flush=True)
        print(f"\t CPU affinity: {list(os.sched_getaffinity(0))}, Host: {socket.gethostname()}", flush=True)
        MPI.COMM_WORLD.Barrier()


    # Problem setup
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
    problem.add_equation("b(z=0) = Lz")
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("b(z=Lz) = 0")
    problem.add_equation("u(z=Lz) = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.sim_time = tBeg
    solver.stop_sim_time = stop_sim_time

    # Initial Conditions
    if initFields is None:
        b.fill_random('g', seed=seed, distribution='normal', scale=1e-3)  # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls
        b['g'] += Lz - z       # Add linear background
    else:
        fields = [(b, "buoyancy"), (p, "pressure"), (u, "velocity"),
                  *[(f, f.name) for f in [tau_p, tau_b1, tau_b2, tau_u1, tau_u2]]]
        for field, name in fields:
            localSlices = (slice(None),) * len(field.tensorsig) + distr.grid_layout.slices(field.domain, field.scales)
            field['g'] = initFields[name][-1][localSlices]

    # Saving snapshots
    if writeFields:
        iterWrite = dtWrite / timestep
        if int(iterWrite) != round(iterWrite, ndigits=3):
            raise ValueError(f"dtWrite ({dtWrite}) is not divisible by timestep ({timestep})")
        iterWrite = int(iterWrite)
        snapshots = solver.evaluator.add_file_handler(
            dirName, sim_dt=dtWrite, max_writes=stop_sim_time / timestep)
        snapshots.add_task(u, name='velocity')
        snapshots.add_task(b, name='buoyancy')
        snapshots.add_task(p, name='pressure')
        if writeFull:
            snapshots.add_task(tau_p, name='tau_p')
            snapshots.add_task(tau_b1, name='tau_b1')
            snapshots.add_task(tau_b2, name='tau_b2')
            snapshots.add_task(tau_u1, name='tau_u1')
            snapshots.add_task(tau_u2, name='tau_u2')
        if writeVort:
            snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    # Main loop
    if nSteps == 0:
        return infos, solver

    try:
        log('Starting main loop')
        t0 = MPI.Wtime()
        for _ in range(nSteps + 1):  # need to do one more step to write last solution ...
            solver.step(timestep)
            if (solver.iteration - 1) % logEvery == 0:
                log(f'Iteration={solver.iteration}, Time={solver.sim_time}, dt={timestep}')
        t1 = MPI.Wtime()
        infos["tComp"] = t1 - t0
        infos["MPI_SIZE"] = MPI_SIZE
        if MPI_RANK == 0:
            with open(f"{dirName}/01_finalized.txt", "w") as f:
                f.write("Done !")
    except:
        log('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

    return infos, solver


if __name__ == '__main__':
    runSim("test", nDim=2)
    runSim("test", nDim=3)

  
import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import numpy as np
import time
from scipy import sparse
import matplotlib.pyplot as plt
from initial_conditions import InvTransSampling
from dynamics import toPeriodic, accelerate, accelerateML, move, push
from field import field
from interpolation import interpMatrix, interpolate
from landau_decay import period, decayRate
from energy import potential


class PICVisualizer:
    def __init__(self, args):
        """
        Visualization utilities for PIC simulations.

        """
        self.args = args
        self.base_dir = Path(__file__).resolve().parent
        self.eval_dir = (self.base_dir / args.evalDir).resolve()
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.QM = args.Qm
        self.N = args.nParticle
        self.NG = args.NG
        self.DT = args.dt
        self.T = args.T
        self.VT = args.Vt
        self.k = args.kc
        self.L = 2*np.pi/self.k                                             # Length of the container
        self.NT = int(self.T/self.DT)                                       # number of time steps
        self.Q = self.L/ (self.QM * self.N)                 # Charge of a particle
        self.dx = self.L / self.NG         
        self.rho_back = - self.Q * self.N / self.L     # background rho                                 # cell length
        self.times = np.linspace(0, self.NT * self.DT, self.NT)
       

        # Set matplotlib defaults (better figures)
        plt.rcParams.update({
            "figure.figsize": (6, 4),
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "grid.alpha": 0.4,
            "grid.linestyle": "--"
        })

    def _img_path(self, name: str) -> Path:
        filename = f"{name}_run{self.args.runId}.{self.args.imgExt}"
        return filename

    def pic1D(self, ml_acc: bool = False, model = None):
        """
        Run a 1D Particle-In-Cell (PIC) simulation.

        Args:
            ml_acc (bool, optional): If True, use machine-learning-based acceleration
                                    instead of the standard PIC acceleration. Default is False.
            model: FNO model Class

        Returns:
            tuple: (xp, vp, wp, E, Ek, Ep, momentum, Exp) where
                xp (np.ndarray): Final particle positions (shape: [N]).
                vp (np.ndarray): Final particle velocities (shape: [N]).
                wp (float or np.ndarray): Particle weights.
                E (list[float]): Total energy per time step.
                Ek (list[float]): Kinetic energy per time step.
                Ep (list[float]): Potential energy per time step.
                momentum (list[float]): Total momentum per time step.
                Exp (list[float]): Electric field energy per time step.

        Notes:
            - Initializes particle positions using inverse transform sampling.
            - Uses quadratic interpolation to project particle charges to the grid.
            - Solves 1D Poisson equation to compute electric potential and field.
            - Updates particle velocities and positions using standard or ML acceleration.
            - Computes kinetic, potential, and field energies, as well as momentum conservation.
        """
        # Storage arrays
        pos = np.zeros([self.NT, self.N], dtype=np.float32)
        Eout = np.zeros([self.NT, self.N], dtype=np.float32)
        p = np.arange(self.N, dtype=int)
        # Build Q-charge  array
        charge = np.full(pos.shape[1], -4 * np.pi, dtype=np.float32)

        # Initial particle positions and velocities
        xt = InvTransSampling(alpha=0.5, k=self.k, L=self.L, N=self.N)
        vp = np.random.randn(self.N)
        wp = 1.0

        # Energy and momentum tracking
        Ek, Ep, Exp, E, momentum = [], [], [], [], []
         # Time tracking
        times_acc = []
     

        for it in range(self.NT):
            # Enforce periodic boundary conditions
            xp = toPeriodic(xt, self.L)

            # Interpolation: particle -> grid
            M = interpMatrix(XP=xp, wp=1, DX=self.dx, N=self.N, NG=self.NG, p=p)
            rho = interpolate(M=M, DX=self.dx, NG=self.NG, Q=self.Q, rho_back=self.rho_back)

            # Compute fields
            phi, Eg = field(rho=rho, L=self.L)

            # Store particle positions
            pos[it, :] = xp.astype(np.float32)
        
            # Acceleration
            if ml_acc and model is not None:
                t0 = time.time()
                # Stack pos and rho
                features = np.stack([pos[it, :], charge], axis=0)
                inputs = features[None, :, :]    # [batch=1, particles, features]
                Eout[it, :] = model(inputs).flatten()
                a = accelerateML(E=Eout, wp=wp, QM=self.QM)
                times_acc.append(time.time() - t0)
            else:
                t0 = time.time()
                a, Eout = accelerate(M=M, E=Eg, Eout=Eout, wp=wp, QM=self.QM, it=it)
                times_acc.append(time.time() - t0)

            # Update velocities and kinetic energy
            vp, kinetic = push(vp=vp, a=a, DT=self.DT, Q=self.Q, QM=self.QM, wp=wp, it=it)

            # Update positions and weights
            xp, wp = move(xp=xp, vp=vp, wp=wp, DT=self.DT, L=self.L, it=it)

            # Compute potential energy
            Epotential = potential(rho=rho, phi=phi, dx=self.dx)

            # Electric field energy
            Egp = np.sum(Eout[it, :] ** 2) * self.L / self.N

            # Append energies and momentum
            Ek.append(kinetic)
            Ep.append(Epotential)
            E.append(kinetic + Epotential)
            Exp.append(Egp)
            momentum.append(np.abs(np.sum(self.Q * vp / self.QM)))

        time_acc_mean = np.round(np.mean(times_acc)*(10**6),3)
        print(f"Average acceleration time per iteration: {time_acc_mean:.3f} microsec")

        return xp, vp, wp, E, Ek, Ep, momentum, Exp, time_acc_mean

    # ---------------------------
    # Plotting methods
    # ---------------------------
    def phase_space(self, xp, vp, wp, ml_acc=False):
        # Filter: only keep particles within velocity cutoff
        mask = np.abs(vp) < 10 * self.VT
        xp = xp[mask]
        vp = vp[mask]

        g1 = np.floor(xp / self.dx).astype(int)
        g = np.array([g1 - 1, g1, g1 + 1])
        g = toPeriodic(g, self.NG, True)

        delta = xp % self.dx
        fraz = np.array([
            (1 - delta) ** 2 / 2,
            1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2),
            delta ** 2 / 2
        ]) * wp

        col = ((vp + 10 * self.VT) // (20 * self.VT / 128)).astype(int)

        n_rows = 128       # velocity bins
        n_cols = self.NG   # grid points

        mat = (
            sparse.csr_matrix((-fraz[0] * self.Q, (col, g[0])), shape=(n_rows, n_cols)) +
            sparse.csr_matrix((-fraz[1] * self.Q, (col, g[1])), shape=(n_rows, n_cols)) +
            sparse.csr_matrix((-fraz[2] * self.Q, (col, g[2])), shape=(n_rows, n_cols))
        ).todense()

        plt.figure(figsize=(7, 5))
        if ml_acc:
            filename = self._img_path("phase_spacePred")
            plt.title("Phase Space Distribution (Pred)")
        else:
            filename = self._img_path("phase_spaceRef")
            plt.title("Phase Space Distribution (Ref)")
        plt.imshow(mat, vmin=0, vmax=np.max(mat), cmap='plasma', interpolation="nearest", aspect='auto')
        plt.colorbar(label="Phase Space Density")
        plt.xlabel("Grid Index")
        plt.ylabel("Velocity Bin")
        plt.tight_layout()
        plt.savefig(f"{self.eval_dir}/{filename}", dpi=200)
        plt.clf()
        return filename

    def energy(self, ERef, EPred=None, EkRef=None, EpRef=None, EkPred=None, EpPred=None):
        plt.figure()
        filename = self._img_path("landau_energy")
        plt.plot(self.times, ERef / ERef[0], label='TotalEnergyRef', color='black')
        if EpPred is not None:
            plt.plot(self.times, EPred / EPred[0], label='TotalEnergyPred', linestyle="--", color='black')
        if EkRef is not None:
            plt.plot(self.times, EkRef / ERef[0], label='KERef', color='blue')
            if EkPred is not None:
                plt.plot(self.times, EkPred / EPred[0], label='KEPred', linestyle="--", color='blue')
        if EpRef is not None:
            plt.plot(self.times, EpRef / ERef[0], label='PERef', color='red')
            if EpPred is not None:
                plt.plot(self.times, EpPred / EPred[0], label='PEPred', linestyle="--", color='red')
        plt.yscale('log')
        plt.ylabel('Normalized Energy')
        plt.xlabel(r'$\omega_p t$')
        plt.legend()
        plt.grid(True)
        plt.title("Landau Energy")
        plt.tight_layout()
        plt.savefig(f"{self.eval_dir}/{filename}", dpi=200)
        plt.clf()
        return filename

    def conservation_errors(self, ERef, pRef, EPred=None, pPred=None):
        plt.figure()
        filename = self._img_path("conservation_errors")
        plt.plot(self.times, np.abs(ERef - ERef[0]) / np.abs(ERef[0]), label='EnergyRef', color='blue')
        plt.plot(self.times, np.abs(pRef - pRef[0]) / np.abs(pRef[0]), label='MomentumRef', color='red')
        if EPred is not None:
            plt.plot(self.times, np.abs(EPred - EPred[0]) / np.abs(EPred[0]), label='EnergyPred', color='blue', linestyle="--")
        if pPred is not None:
            plt.plot(self.times, np.abs(pPred - pPred[0]) / np.abs(pPred[0]), label='MomentumPred', color='red', linestyle="--")
        plt.yscale('log')
        plt.ylabel('Relative Error')
        plt.xlabel(r'$\omega_p t$')
        plt.title("Error in Energy and Momentum Conservation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.eval_dir}/{filename}", dpi=200)
        plt.clf()
        return filename

    def landau_decay(self, phiMax, phiMaxPred=None):
        a = np.linspace(0, (self.NT - 1) * self.DT, self.NT)
        pp = period(self.k)
        b = phiMax[int(pp // (2 * self.DT))] * np.exp((a[0:2000] - pp / 2) * decayRate(self.k))
        plt.figure()
        filename = self._img_path("landau_decay_rateRef")
        plt.plot(a, phiMax, label=r'$\int ERef_x^2 dV$', color='blue')
        plt.plot(a[0:2000], b, label='Predicted Decay Rate', color='green', linestyle="--")
        if phiMaxPred is not None:
            plt.plot(a, phiMaxPred, label=r'$\int EPred_x^2 dV$', color='orange')
        plt.title(f'Landau Damping Decay Rate (k={self.k})')
        plt.yscale('log')
        plt.ylabel(r'$\int E_x^2 dV$')
        plt.xlabel(r'$\omega_p t$')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.eval_dir}/{filename}", dpi=200)
        plt.clf()
        return filename

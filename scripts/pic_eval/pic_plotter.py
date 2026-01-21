import sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))
import numpy as np
import cupy as cp
import time
from scipy import sparse
import matplotlib.pyplot as plt
import h5py
from initial_conditions import InvTransSampling
from dynamics import toPeriodic, accelerate, accelerateML, move, push, toPeriodicND, toPeriodicNDTranspose
from field import field
from interpolation import interpMatrix, interpolate
from landau_decay import period, decayRate
from energy import potential
from operator_learning.data.pic_dataset import normalize_per_sample

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
        self.NT = int(self.T/self.DT) 
        self.times = np.linspace(0, self.NT * self.DT, self.NT)   # number of time steps
        self.dim = args.dim
        self.k = np.array([args.kc]*self.dim)
        self.L = 2*np.pi/self.k                                   # Length of the container  
        self.dx = np.array(self.L / self.NG) # cell length 
        self.Ln = cp.asarray(self.L)
        self.dxn = cp.asarray(self.dx)
        self.alpha = args.alpha
        
        if self.dim == 1:                                                             
            self.Q = self.L[0]/ (self.QM * self.N)                                 # Charge of a particle                                                    
            self.rho_back = - self.Q * self.N / self.L[0]                          # background rho     
        else:
            self.Q = self.L[0] * self.L[1] / (self.QM * self.N)  
            self.rho_back = - self.Q * self.N / (self.L[0] * self.L[1])
       

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

    def pic1D(self, ml_acc: bool = False, model = None, data_file = None):
        """
        Run a 1D Particle-In-Cell (PIC) simulation.

        Args:
            ml_acc (bool, optional): If True, use machine-learning-based acceleration
                                    instead of the standard PIC acceleration. Default is False.
            model: FNO model Class

        Returns:
            tuple: (xp, vp, wp, E, Ek, Ep, momentum, Exp) where
                xp (cp.ndarray): Final particle positions (shape: [N]).
                vp (cp.ndarray): Final particle velocities (shape: [N]).
                wp (float or cp.ndarray): Particle weights.
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
        pos = cp.zeros([self.NT, self.N], dtype=cp.float32)
        Eout = cp.zeros([self.NT, self.N], dtype=cp.float32)
        p = cp.arange(self.N, dtype=int)
        # Build Q-charge  array
        charge = cp.full(pos.shape[1], -4 * cp.pi, dtype=cp.float32)

        # Initial particle positions and velocities
        xp = InvTransSampling(alpha=self.alpha, k=self.k, L=self.L, N=self.N, dim=self.dim)
        xp = cp.asarray(xp)
        vp = cp.random.randn(self.N)
        wp = 1.0

        # Mean and Std of training output data
        if data_file is not None:
            data = h5py.File(data_file, 'r')
            data_output_mean = data['infos']['output_mean'][()]
            data_output_std = data['infos']['output_std'][()]


        # Energy and momentum tracking
        Ek, Ep, Exp, E, momentum = [], [], [], [], []
         # Time tracking
        times_acc = []
     

        print('Before entering time loop')
        for it in range(self.NT):
            print(it)
            # Enforce periodic boundary conditions
            xp = toPeriodic(xp, self.Ln)
            #breakpoint()
            # Store particle positions
            pos[it, :] = xp.astype(cp.float32)
        
            # Acceleration
            if ml_acc and model is not None:
                t0 = time.time()
                # Stack pos and rho
                features = cp.stack([pos[it, :], charge], axis=0)
                inputs = features[None, :, :]    # [batch=1, particles, features]
                inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
                #out = model(inputs).flatten()
                #breakpoint()
                Eout[it, :] = model(inputs).flatten()
                Eout[it,:] = Eout[it,:] * data_output_std + data_output_mean
                #breakpoint()
                a = accelerateML(E=Eout[it,:].squeeze(), wp=wp, QM=self.QM)
                #breakpoint()
                times_acc.append(time.time() - t0)
            else:
                t0 = time.time()
                # Interpolation: particle -> grid
                M = interpMatrix(XP=xp, wp=1, DX=self.dxn, N=self.N, NG=self.NG, p=p, L=self.Ln, dim=self.dim)
                rho = interpolate(M=M, DX=self.dxn, L=self.Ln, NG=self.NG, Q=self.Q, rho_back=self.rho_back, dim=self.dim)

                # Compute fields
                phi, Eg = field(rho=rho, L=self.Ln, dim=self.dim)
                
                a, Eout = accelerate(M=M, E=Eg, Eout=Eout, wp=wp, QM=self.QM, it=it, dim=self.dim)
                times_acc.append(time.time() - t0)

            # Update velocities and kinetic energy
            vp, kinetic = push(vp=vp, a=a, DT=self.DT, Q=self.Q, QM=self.QM, wp=wp, it=it)

            #breakpoint()
            # Update positions and weights
            xp, wp = move(xp=xp, vp=vp, wp=wp, DT=self.DT, L=self.Ln, it=it)
            #breakpoint()

            # Electric field energy
            Egp = cp.sum(Eout[it, :] ** 2) * self.Ln / self.N

            # Compute potential energy
            # Epotential = potential(rho=rho, phi=phi, dx=self.dx, dim=self.dim)
            Epotential = 0.5 * Egp
            
            # Append energies and momentum
            Ek.append(kinetic.get())
            Ep.append(Epotential.get())
            E.append((kinetic + Epotential).get())
            Exp.append(Egp.get())
            momentum.append((cp.abs(cp.sum(self.Q * vp / self.QM))).get())

        time_acc_mean = np.round(np.mean(times_acc)*(10**6),3)
        print(f"Average acceleration time per iteration: {time_acc_mean:.3f} microsec")

        return xp, vp, wp, E, Ek, Ep, momentum, Exp, time_acc_mean

    def pic2D(self, ml_acc: bool = False, model = None, data_file = None):
        # Storage arrays
        pos = cp.zeros([self.NT, self.N, 2], dtype=cp.float32)
        Eout = cp.zeros([self.NT, self.N, 2], dtype=cp.float32)
        p = cp.arange(self.N, dtype=int)
        # Build Q-charge  array
        charge = cp.full((1, 1, pos.shape[1]), self.Q*self.N, dtype=cp.float32)

        # Initial particle positions and velocities
        xpn, vpn = InvTransSampling(alpha=self.alpha, k=self.k, L=self.L, N=self.N, dim=self.dim)
        xpc, vpc = np.transpose(xpn), np.transpose(vpn)
        xp, vp = cp.asarray(xpc), cp.asarray(vpc)
        xpn, vpn = cp.asarray(xpn), cp.asarray(vpn)
        wp = 1.0

        # Mean and Std of training output data
        if data_file is not None:
            data = h5py.File(data_file, 'r')
            data_output_mean = data['infos']['output_mean'][()]
            data_output_std = data['infos']['output_std'][()]

        # Energy and momentum tracking
        Ek, Ep, Exp, E, momentum = [], [], [], [], []
         # Time tracking
        times_acc = []
     
        for it in range(self.NT):
           
            # Acceleration
            if ml_acc and model is not None:
                t0 = time.time()
                # Enforce periodic boundary conditions
                xp = toPeriodicND(xp, self.L, dim=self.dim) # [particle, channel]
                # Stack pos and charge
                xt = cp.transpose(xp)  # [channel, particle]
                positions = xt[None, :, :]
                positions[:, 0, :] = normalize_per_sample(positions[:, 0, :])
                positions[:, 1, :] = normalize_per_sample(positions[:, 1, :])
                inputs = cp.concatenate([positions, charge], axis=1) # [batch=1, channel=3, particles]
                prediction = model(inputs) # [1, channel=2, particle]
                Efieldparticle = cp.swapaxes(prediction, 1, 2).squeeze()  # [particles, channel=2]
                Efieldparticle[:,0] = Efieldparticle[:,0] * data_output_std[0] + data_output_mean[0]
                Efieldparticle[:,1] = Efieldparticle[:,1] * data_output_std[1] + data_output_mean[1]
                Efieldparticle[:,0] = Efieldparticle[:,0] - ((1/N) * cp.sum(Efieldparticle[:,0]))
                Efieldparticle[:,1] = Efieldparticle[:,1] - ((1/N) * cp.sum(Efieldparticle[:,1]))
                a = accelerateML(E=Efieldparticle, wp=wp, QM=self.QM)
                times_acc.append(time.time() - t0)
                vp, kinetic = push(vp=vp, a=a, DT=self.DT, Q=self.Q, QM=self.QM, wp=wp, it=it)
                # Update positions and weights
                xp, wp = move(xp=xp, vp=vp, wp=wp, DT=self.DT, L=self.L, it=it)
            else:
                t0 = time.time()
                xpn = toPeriodicNDTranspose(xpn, self.L, dim=self.dim)
                # Interpolation: particle -> grid
                M = interpMatrix(XP=xpn, wp=1, DX=self.dx, N=self.N, NG=self.NG, p=p, L=self.L, dim=self.dim)
                rho = interpolate(M=M, DX=self.dx, L=self.L, NG=self.NG, Q=self.Q, rho_back=self.rho_back, dim=self.dim)

                # Compute fields
                phi, Eg = field(rho=rho, L=self.L, dim=self.dim)
                pos[it, :, :] = cp.transpose(xpn.astype(cp.float32))
                an, Eout = accelerate(M=M, E=Eg, Eout=Eout, wp=wp, QM=self.QM, it=it, dim=self.dim)
                Efieldparticle = Eout[it,:,:].squeeze()
                times_acc.append(time.time() - t0)
                # Update velocities and kinetic energy
                vpn, kineticn = push(vp=vpn, a=an, DT=self.DT, Q=self.Q, QM=self.QM, wp=wp, it=it)
                # Update positions and weights
                xpn, wp = move(xp=xpn, vp=vpn, wp=wp, DT=self.DT, L=self.L, it=it)



            # Electric field energy
            Egp = cp.sum(Efieldparticle[:,1]**2) * (self.L[0] * self.L[1]) / self.N
   
            # Compute potential energy
            Epotential = cp.sum(Efieldparticle[:,0]**2 + Efieldparticle[:,1]**2) * 0.5 * (self.L[0] * self.L[1]) / self.N
            
            # Append energies and momentum
            Ek.append(kinetic.get())
            Ep.append(Epotential.get())
            E.append((kinetic + Epotential).get())
            Exp.append(Egp.get())
            momx = cp.sum(self.Q * vp[:,0] / self.QM)
            momy = cp.sum(self.Q * vp[:,1] / self.QM)
            momentum.append((cp.sqrt(momx**2 + momy**2)).get())

        time_acc_mean = np.round(np.mean(times_acc)*(10**3),3)
        print(f"Average acceleration time per iteration: {time_acc_mean:.3f} millisec")

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

    def landau_decay(self, Ex, ExPred=None, label='weak'):
        a = np.linspace(0, (self.NT - 1) * self.DT, self.NT)
        #pp = period(self.k[0])
        #b = phiMax[int(pp // (2 * self.DT))] * np.exp((a[0:2000] - pp / 2) * decayRate(self.k[0]))
        plt.figure()
        filename = self._img_path("landau_decay_rateRef")
        plt.plot(a, Ex, label=r'$\int ERef_x^2 dV$', color='blue')
        #plt.plot(a[0:2000], b, label='Predicted Decay Rate', color='green', linestyle="--")
        if(label == 'weak'):
            gamma1 = -0.3066
        else:
            gamma1 = -0.562
            gamma2 = 0.168
            ind2 = np.argmin(np.abs(a - 20.592))
            theo_ref2 = np.exp(gamma2 * a)
            theo_ref2 = (Ex[ind2]/theo_ref2[ind2])*theo_ref2


        ind1 = np.argmin(np.abs(a - 2.5))
        theo_ref1 = np.exp(gamma1 * a)
        theo_ref1 = (Ex[ind1]/theo_ref1[ind1])*theo_ref1
        plt.plot(a, theo_ref1, label='Predicted Decay Rate', color='seagreen')
        if(label == 'strong'):
            plt.plot(a, theo_ref2, label='Predicted Growth Rate', color='red')
        if ExPred is not None:
            plt.plot(a, ExPred, label=r'$\int EPred_x^2 dV$', color='orange')
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

    def visualize_spatial_comparison(self, true_positions, predicted_positions, true_efield, predicted_efield, timestep, output_filename):
        """
        Creates a 2x2 grid of plots showing particle positions colored by E-field values.
        Top row compares Ex (Predicted vs. True), Bottom row compares Ey.
        Crucially, it uses a shared color scale for each row to allow direct comparison.
        """
        # Use a diverging colormap, which is great for fields (positive/negative values)
        cmap = 'coolwarm'
        marker_size = 1 # Use a small marker size for 100k points

        # --- 1. Determine the shared color range for Ex and Ey ---
        # For Ex, find the min/max across both predicted and true values
        vmin_x = min(predicted_efield[:, 0].min(), true_efield[:, 0].min())
        vmax_x = max(predicted_efield[:, 0].max(), true_efield[:, 0].max())
        
        # For Ey, do the same
        vmin_y = min(predicted_efield[:, 1].min(), true_efield[:, 1].min())
        vmax_y = max(predicted_efield[:, 1].max(), true_efield[:, 1].max())

        # --- 2. Create the 2x2 plot grid ---
        fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=150)
        fig.suptitle(f'Spatial E-Field Comparison for Timestep {timestep}', fontsize=20)

        # --- 3. Plot the Ex comparison (Top Row) ---
        # Predicted Ex
        sc00 = axes[0, 0].scatter(predicted_positions[:, 0], predicted_positions[:, 1], c=predicted_efield[:, 0], 
                                cmap=cmap, vmin=vmin_x, vmax=vmax_x, s=marker_size, rasterized=True)
        axes[0, 0].set_title('Predicted $E_x$', fontsize=14)
        
        # True Ex
        sc01 = axes[0, 1].scatter(true_positions[:, 0], true_positions[:, 1], c=true_efield[:, 0], 
                                cmap=cmap, vmin=vmin_x, vmax=vmax_x, s=marker_size, rasterized=True)
        axes[0, 1].set_title('Ground Truth $E_x$', fontsize=14)

        # Add a single colorbar for the Ex row
        fig.colorbar(sc01, ax=axes[0, :], label='$E_x$ Value', fraction=0.046, pad=0.04)

        # --- 4. Plot the Ey comparison (Bottom Row) ---
        # Predicted Ey
        sc10 = axes[1, 0].scatter(predicted_positions[:, 0], predicted_positions[:, 1], c=predicted_efield[:, 1], 
                                cmap=cmap, vmin=vmin_y, vmax=vmax_y, s=marker_size, rasterized=True)
        axes[1, 0].set_title('Predicted $E_y$', fontsize=14)

        # True Ey
        sc11 = axes[1, 1].scatter(true_positions[:, 0], true_positions[:, 1], c=true_efield[:, 1], 
                                cmap=cmap, vmin=vmin_y, vmax=vmax_y, s=marker_size, rasterized=True)
        axes[1, 1].set_title('Ground Truth $E_y$', fontsize=14)
        
        # Add a single colorbar for the Ey row
        fig.colorbar(sc11, ax=axes[1, :], label='$E_y$ Value', fraction=0.046, pad=0.04)

        # --- 5. Final plot adjustments ---
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel('Particle X Position')
                ax.set_ylabel('Particle Y Position')
                ax.set_aspect('equal', adjustable='box') # Ensure correct spatial aspect ratio
                ax.grid(True, linestyle='--', alpha=0.5)
        filename = self._img_path(output_filename)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.eval_dir}/{filename}', dpi=200)
        plt.close()
  
    def twoStreamIppl(self, ExRef, ExPred = None, label='tsi'):
        a = np.linspace(0, (self.NT - 1) * self.DT, self.NT)
        plt.plot(a, ExRef, label=r'$\int E_{x}Ref^2 dV$')
        if ExPred is not None:
            plt.plot(a, ExPred, label=r'$\int E_{x}Pred^2 dV$')
        if(label == 'tsi'):
            gamma = 0.4952
        else:
            gamma = 0.356
        filename = self._img_path("growth_rate")
        ind = np.argmin(np.abs(a - 8.0))
        theo_ref = np.exp(gamma * a)
        theo_ref = (ExRef[ind]/theo_ref[ind])*theo_ref
        plt.plot(a, theo_ref, label='predicted growth rate', color='seagreen')
        # plt.title('Landau Damping Decay Rate, k=0.5', fontsize='14')
        plt.yscale('log')
        ax = plt.gca()
        if(label == 'tsi'):
            ax.set_ylim([1e-4,1e3])
        else:
            ax.set_ylim([1e-2,1e3])
        plt.ylabel(r'$\int E_x^2 dV$', fontsize='14')
        plt.xlabel(r'normalized time unit: $\omega_p$t', fontsize='14')
        plt.legend()
        plt.grid(color='gray')
        plt.savefig(f'{self.eval_dir}/{filename}', dpi=200)
        plt.clf()
        return filename

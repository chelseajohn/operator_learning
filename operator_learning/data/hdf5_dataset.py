import h5py
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from operator_learning.data.datasets.rbc.dedalus_prop import OutputFiles
from operator_learning.utils.misc import print_rank0

class HDF5Dataset(Dataset):

    def __init__(self, dataFile, **kwargs):
        """
        Dataset reader and getitem for DataLoader

        Args:
            dataFile (hdf5): data file 
            
        """
        
        self.file = h5py.File(dataFile, 'r')
        self.inputs = self.file['inputs']
        self.outputs = self.file['outputs']
        self.nDim = kwargs.get('space_dim', 2)


        if self.nDim == 2:
            xGrid, yGrid = self.grid
        else:
            xGrid, yGrid, zGrid = self.grid
            self.nZ = zGrid.size
            self.kY = kwargs.get('kZ', 12)

        self.nX = xGrid.size
        self.nY = yGrid.size
        self.kX = kwargs.get('kX', 12)
        self.kY = kwargs.get('kY', 12)
  
        assert len(self.inputs) == len(self.outputs), \
            f"different sample number for inputs and outputs ({len(self.inputs)},{len(self.outputs)})"
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inpt, outp = self.sample(idx)
        return torch.tensor(inpt), torch.tensor(outp)

    def __del__(self):
        try:
            self.file.close()
        except:
            pass

    def sample(self, idx):
        return self.inputs[idx], self.outputs[idx]

    @property
    def infos(self):
        return self.file["infos"]

    @property
    def grid(self):
        if self.nDim == 2:
            return self.infos["xGrid"][:], self.infos["yGrid"][:]
        else:
            return self.infos["xGrid"][:], self.infos["yGrid"][:], self.infos["zGrid"][:]

    @property
    def outType(self):
        return self.infos["outType"][()].decode("utf-8")

    @property
    def outScaling(self):
        return float(self.infos["outScaling"][()])

    def calc_minSlice(self, n, modes):
        """
        Finding min number of points to satisfy
        n/2 + 1 >= fourier modes
        """
        slice_min = 2*(modes-1)
        if slice_min < n:
            return slice_min
        else:
            print_rank0("Insufficient number of points to slice")
            return 0

    def printInfos(self):
        print_rank0(f"### Dataset Infos ###")
        if self.nDim == 2:
            xGrid, yGrid = self.grid
            print_rank0(f" -- grid shape : ({xGrid.size}, {yGrid.size})")
            print_rank0(f" -- grid domain : [{xGrid.min():.1f}, {xGrid.max():.1f}] x [{yGrid.min():.1f}, {yGrid.max():.1f}]")
        else:
            xGrid, yGrid, zGrid = self.grid
            print_rank0(f" -- grid shape : ({xGrid.size}, {yGrid.size}, {zGrid.size})")
            print_rank0(f" -- grid domain : [{xGrid.min():.1f}, {xGrid.max():.1f}] x [{yGrid.min():.1f}, \
                                       {yGrid.max():.1f}] x [{zGrid.min():.1f}, {zGrid.max():.1f}]")
        infos = self.infos
        print_rank0(f" -- nSimu : {infos['nSimu'][()]}")
        print_rank0(f" -- dtData : {infos['dtData'][()]:1.2g}")
        print_rank0(f" -- inSize : {infos['inSize'][()]}")                # T_in
        print_rank0(f" -- outStep : {infos['outStep'][()]}")              # T
        print_rank0(f" -- inStep : {infos['inStep'][()]}")                # tStep
        print_rank0(f" -- nSamples (per simu) : {infos['nSamples'][()]}")
        print_rank0(f" -- nSamples (total) : {infos['nSamples'][()]*infos['nSimu'][()]}")
        print_rank0(f" -- dtInput : {infos['dtInput'][()]:1.2g}")
        print_rank0(f" -- outType : {infos['outType'][()].decode('utf-8')}")
        print_rank0(f" -- outScaling : {infos['outScaling'][()]:1.2g}")

class DomainDataset(HDF5Dataset):
    """
    A dataset class that supports different sampling strategies: 
    'random', 'fixed', or 'ordered' for 2-D grid data.

    Args:
        dataFile (hdf5): Input HDF5 file.
        sampling_mode (str): 'random', 'fixed', or 'ordered'.
                             'ordered': To divide full grid (nX,nY) into (nX//sX)*(nY//sY) 
                                        exactly divisible (sX,sY) size patches w/o overlapping.
                             'random':  To divide full grid (nX, nY) into random (sX,sY) 
                                        size patches with (possible) overlapping.
                             'fixed':   To divide full grid (nX,nY) into fixed (sX, sY) size 
                                        patches with (possible) overlapping.
        pad_to_fullGrid (bool): Pad (sX,sY) into (nX,nY) grid with zeros. Default: False.
        nPatch_per_sample (int): Number of patches per sample. Default: 1.
        use_minLimit (bool): Restrict (sX,sY) > (2*kX-1, 2*kY-1). Default: True.
        use_fixedPatch_startIdx (bool): To divide full grid (nX,nY) into nPatch_per_sample 
                                        (sX,sY) sized patches starting from same index
                                        per epoch. Default: True.
        padding (list): Padding format [left, right, bottom, top]. Default: [0, 0, 0, 0].
        slices (list): List of patch sizes (sX,sY). Default: [].
        patch_startIdx (list): Starting index for each patch. Default: [[0,0]].
        kX, kY (int): Fourier modes used to compute minimum patch size.
    """
    def __init__(self, dataFile,
                 sampling_mode='random',  # 'random', 'fixed', 'ordered'
                 pad_to_fullGrid=False,
                 nPatch_per_sample=1,
                 use_minLimit=True,
                 use_fixedPatch_startIdx=True,
                 padding=[0, 0, 0, 0],
                 **kwargs):

        super().__init__(dataFile, **kwargs)
        self.sampling_mode = sampling_mode
        self.nPatch_per_sample = nPatch_per_sample
        self.pad_to_fullGrid = pad_to_fullGrid
        self.use_fixedPatch_startIdx = use_fixedPatch_startIdx
        self.use_minLimit = use_minLimit or not pad_to_fullGrid
        self.padding = padding # [left, right, bottom, top]

        slices = kwargs.get('slices', [])
        if not slices:
            single_slice = self.find_patchSize()
        else:
            single_slice = slices

        if sampling_mode == 'ordered':
            assert len(single_slice) == 1, f"{len(single_slice)} patch sizes given for ordered sampling"
            sX, sY = single_slice[0]
            self.nPatch_per_sample = (self.nX // sX) * (self.nY // sY)
            self.slices = single_slice * self.nPatch_per_sample
        elif sampling_mode == 'fixed':
            self.slices = single_slice * self.nPatch_per_sample
        elif sampling_mode == 'random':
            self.slices = single_slice
            assert len(self.slices) == self.nPatch_per_sample, "Mismatch in slices and nPatch_per_sample"
        else:
            raise ValueError(f"Invalid sampling_mode: {sampling_mode}")

        assert not (use_fixedPatch_startIdx and sampling_mode == 'ordered'), \
            "use_fixedPatch_startIdx and ordered sampling cannot be True at the same time."

        patch_startIdx = kwargs.get('patch_startIdx', [])
        if use_fixedPatch_startIdx:
            if len(patch_startIdx) == len(self.slices):
                self.patch_startIdx = patch_startIdx
            else:
                self.patch_startIdx = self.find_patch_startIdx()

    def __getitem__(self, idx):
        patch_padding = self.padding.copy()
        iSample = idx // self.nPatch_per_sample
        iPatch = idx % self.nPatch_per_sample
        inpt_grid, outp_grid = self.sample(iSample)
        sX, sY = self.slices[iPatch]

        if self.use_fixedPatch_startIdx:
            xPatch_startIdx, yPatch_startIdx = self.patch_startIdx[iPatch]
        elif self.sampling_mode == 'ordered':
            if sX == self.nX and sY == self.nY:
                xPatch_startIdx = 0
                yPatch_startIdx = 0
            else:
                nX_div, nY_div = self.nX // sX, self.nY // sY
                xPatch_startIdx = (iPatch % nX_div) * sX
                yPatch_startIdx = (iPatch % nY_div) * sY
        else:  # 'random'
            xPatch_startIdx = random.randint(0, self.nX - sX)
            yPatch_startIdx = random.randint(0, self.nY - sY)

        # Adjust patch padding
        patch_padding[0] = 0 if xPatch_startIdx == 0 or (xPatch_startIdx - patch_padding[0]) < 0 else patch_padding[0]
        patch_padding[1] = 0 if (xPatch_startIdx + sX + patch_padding[1]) >= self.nX else patch_padding[1]
        patch_padding[2] = 0 if yPatch_startIdx == 0 or (yPatch_startIdx - patch_padding[2]) < 0 else patch_padding[2]
        patch_padding[3] = 0 if (yPatch_startIdx + sY + patch_padding[3]) >= self.nY else patch_padding[3]

        x0, x1 = xPatch_startIdx - patch_padding[0], xPatch_startIdx + sX + patch_padding[1]
        y0, y1 = yPatch_startIdx - patch_padding[2], yPatch_startIdx + sY + patch_padding[3]

        if self.pad_to_fullGrid:
            inpt = np.zeros_like(inpt_grid)
            outp = np.zeros_like(outp_grid)
            inpt[:, :x1 - x0, :y1 - y0] = inpt_grid[:, x0:x1, y0:y1]
            outp[:, :x1 - x0, :y1 - y0] = outp_grid[:, x0:x1, y0:y1]
        else:
            inpt = inpt_grid[:, x0:x1, y0:y1]
            outp = outp_grid[:, x0:x1, y0:y1]

        return torch.tensor(inpt), torch.tensor(outp)

    def find_patchSize(self):
        slices = []
        nX_min, nY_min = (self.calc_sliceMin(self.nX, self.kX), self.calc_sliceMin(self.nY, self.kY)) if self.use_minLimit else (1, 1)
        if self.sampling_mode == 'ordered':
            self.valid_sX = [sx for sx in range(nX_min, self.nX) if self.nX % sx == 0]
            self.valid_sY = [sy for sy in range(nY_min, self.nY) if self.nY % sy == 0]
            # select a (sX,sY) randomly 
            sX = int(random.choice(self.valid_sX))
            sY = int(random.choice(self.valid_sY))
            slices.append((sX, sY))
        else:
            for _ in range(self.nPatch_per_sample):
                sX = random.randint(nX_min, self.nX)
                sY = random.randint(nY_min, self.nY)
                slices.append((sX, sY))
        return slices

    def find_patch_startIdx(self):
        patch_start = []
        for sX, sY in self.slices:
            xPatch_startIdx = random.randint(0, self.nX - sX)
            yPatch_startIdx = random.randint(0, self.nY - sY)
            patch_start.append((xPatch_startIdx, yPatch_startIdx))
        return patch_start

    def printInfos(self):
        xGrid, yGrid = self.grid
        infos = self.infos
        print_rank0(f" -- grid shape : ({xGrid.size}, {yGrid.size})")
        print_rank0(f" -- grid domain : [{xGrid.min():.1f}, {xGrid.max():.1f}] x [{yGrid.min():.1f}, {yGrid.max():.1f}]")
        print_rank0(f" -- nSimu : {infos['nSimu'][()]}")
        print_rank0(f" -- dtData : {infos['dtData'][()]:1.2g}")
        print_rank0(f" -- inSize : {infos['inSize'][()]}")          # T_in
        print_rank0(f" -- outStep : {infos['outStep'][()]}")        # T
        print_rank0(f" -- inStep : {infos['inStep'][()]}")          # tStep
        print_rank0(f" -- nSamples (per simu) : {infos['nSamples'][()]}")
        print_rank0(f" -- nSamples (total) : {infos['nSamples'][()] * infos['nSimu'][()]}")
        print_rank0(f" -- dtInput : {infos['dtInput'][()]:1.2g}")
        print_rank0(f" -- outType : {infos['outType'][()].decode('utf-8')}")
        print_rank0(f" -- outScaling : {infos['outScaling'][()]:1.2g}")
        print_rank0(f" -- sampling_mode: {self.sampling_mode}")
        print_rank0(f" -- pad_to_fullGrid: {self.pad_to_fullGrid}")
        print_rank0(f" -- nPatch (per sample): {self.nPatch_per_sample}")
        print_rank0(f" -- patches (per sample): {self.slices}")
        print_rank0(f" -- padding (per patch): {self.padding}")
        if self.use_minLimit:
            print_rank0(f"Min nX & nY for patch computed using (kX={self.kX}, kY={self.kY})")
        if self.use_fixedPatch_startIdx:
            print_rank0(f" -- patch start index (per epoch): {self.patch_startIdx}")

def createDataset(
        dataDir, inSize, outStep, inStep, outType, outScaling, dataFile,
        verbose=False, nDim=2, **kwargs):
    assert inSize == 1, "inSize != 1 not implemented yet ..."
    simDirsSorted = sorted(glob.glob(f"{dataDir}/simu_*"), key=lambda f: int(f.split('simu_',1)[1]))
    nSimu = int(kwargs.get("nSimu", len(simDirsSorted)))
    simDirs = simDirsSorted[:nSimu]
    print_rank0('Using Simulations:')
    for s in simDirs:
        print_rank0(f" -- {s}")

    # -- retrieve informations from first simulation
    outFiles = OutputFiles(f"{simDirs[0]}/run_data")
    nFields = sum(outFiles.nFields)
    fieldShape = outFiles.shape
    times = outFiles.times().ravel()
    if nDim == 2:
        xGrid, yGrid = outFiles.x, outFiles.y  # noqa: F841 (used lated by an eval call)
    else:
        xGrid, yGrid, zGrid = outFiles.x, outFiles.y, outFiles.z # noqa: F841 (used lated by an eval call)
    
    dtData = times[1]-times[0]
    dtInput = dtData*outStep  # noqa: F841 (used lated by an eval call)
    dtSample = dtData*inStep  # noqa: F841 (used lated by an eval call)
    
    iBeg = int(kwargs.get("iBeg", 0))
    iEnd = int(kwargs.get("iEnd", nFields))
    sRange = range(iBeg, iEnd-inSize-outStep+1, inStep)
    nSamples = len(sRange)
    print_rank0(f'selector: {sRange},  outStep: {outStep}, inStep: {inStep}, iBeg: {iBeg}, iEnd: {iEnd}')

    infoParams = [
        "inSize", "outStep", "inStep", "outType", "outScaling", "iBeg", "iEnd",
        "dtData", "dtInput", "xGrid", "yGrid", "nSimu", "nSamples", "dtSample",
    ]
    
    if nDim == 3:
        infoParams += ["zGrid"]

    print_rank0(f"Creating dataset from {nSimu} simulations, {nSamples} samples each ...")
    dataset = h5py.File(dataFile, "w")
    for name in infoParams:
        try:
            dataset.create_dataset(f"infos/{name}", data=np.asarray(eval(name)))
        except:
            dataset.create_dataset(f"infos/{name}", data=eval(name))

    dataShape = (nSamples*nSimu, *fieldShape)
    print_rank0(f'data shape: {dataShape}')
    inputs = dataset.create_dataset("inputs", dataShape)
    outputs = dataset.create_dataset("outputs", dataShape)
    for iSim, dataDir in enumerate(simDirs):
        outFiles = OutputFiles(f"{dataDir}/run_data")
        print_rank0(f" -- sampling data from {dataDir}/run_data")
        for iSample, iField in enumerate(sRange):
            if verbose:
                print_rank0(f"\t -- creating sample {iSample+1}/{nSamples}")
            inpt, outp = outFiles.fields(iField), outFiles.fields(iField+outStep).copy()
            if outType == "update":
                outp -= inpt
                if outScaling != 1:
                    outp *= outScaling
            inputs[iSim*nSamples + iSample] = inpt
            outputs[iSim*nSamples + iSample] = outp
    dataset.close()
    print_rank0(" -- done !")
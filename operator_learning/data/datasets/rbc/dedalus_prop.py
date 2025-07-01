import os
import h5py
import glob
import random
import numpy as np
from qmat.lagrange import LagrangeApproximation

def getModes(grid):
    nX = np.size(grid)
    k = np.fft.rfftfreq(nX, 1/nX) + 0.5
    return k

def decomposeRange(iBeg, iEnd, step, maxSize):
    if iEnd is None:
        raise ValueError("need to provide iEnd for range decomposition")
    nIndices = len(range(iBeg, iEnd, step))
    subRanges = []

    # Iterate over the original range and create sub-ranges
    iStart = iBeg
    while nIndices > 0:
        iStop = iStart + (maxSize - 1) * step
        if step > 0 and iStop > iEnd:
            iStop = iEnd
        elif step < 0 and iStop < iEnd:
            iStop = iEnd

        subRanges.append((iStart, iStop + 1 - (iStop==iEnd), step))
        nIndices -= maxSize
        iStart = iStop + step if nIndices > 0 else iEnd

    return subRanges

def computeMeanSpectrum(uValues, xGrid=None, zGrid=None, verbose=False):
    """ uValues[nT, nVar, nX, (nY), nZ] """
    uValues = np.asarray(uValues)
    nT, nVar, *gridSizes = uValues.shape
    dim = len(gridSizes)
    # assert nVar == dim
    if verbose:
        print(f"Computing Mean Spectrum on u[{', '.join([str(n) for n in uValues.shape])}]")

    energy_spectrum = []
    if dim == 2:

        for i in range(2):
            u = uValues[:, i]                           # (nT, Nx, Nz)
            spectrum = np.fft.rfft(u, axis=-2)          # over Nx -->  #(nT, k, Nz)
            spectrum *= np.conj(spectrum)               # (nT, k, Nz)
            spectrum /= spectrum.shape[-2]              # normalize with Nx --> (nT, k, Nz)
            spectrum = np.mean(spectrum.real, axis=-1)  # mean over Nz --> (nT,k)
            energy_spectrum.append(spectrum)

    elif dim == 3:

        # Check for a cube with uniform dimensions
        nX, nY, nZ = gridSizes
        assert nX == nY
        size = nX // 2

        # Interpolate in z direction
        assert xGrid is not None and zGrid is not None
        if verbose: print(" -- interpolating from zGrid to a uniform mesh ...")
        from qmat.lagrange import LagrangeApproximation
        P = LagrangeApproximation(zGrid).getInterpolationMatrix(xGrid)
        np.einsum('ij,tvxyj->tvxyi', P, uValues, out=uValues)

        # Compute 3D mode shells
        k1D = np.fft.fftfreq(nX, 1/nX)**2
        kMod = k1D[:, None, None] + k1D[None, :, None] + k1D[None, None, :]
        kMod **= 0.5
        idx = kMod.copy()
        idx *= (kMod < size)
        idx -= (kMod >= size)

        idxList = range(int(idx.max()) + 1)
        flatIdx = idx.ravel()

        # Fourier transform and square of Im,Re
        if verbose: print(" -- 3D FFT on u, v & w ...")
        uHat = np.fft.fftn(uValues, axes=(-3, -2, -1))

        if verbose: print(" -- square of Im,Re ...")
        ffts = [uHat[:, i] for i in range(nVar)]
        reParts = [uF.reshape((nT, nX*nY*nZ)).real**2 for uF in ffts]
        imParts = [uF.reshape((nT, nX*nY*nZ)).imag**2 for uF in ffts]

        # Spectrum computation
        if verbose: print(" -- computing spectrum ...")
        spectrum = np.zeros((nT, size))
        for i in idxList:
            if verbose: print(f" -- k{i+1}/{len(idxList)}")
            kIdx = np.argwhere(flatIdx == i)
            tmp = np.empty((nT, *kIdx.shape))
            for re, im in zip(reParts, imParts):
                np.copyto(tmp, re[:, kIdx])
                tmp += im[:, kIdx]
                spectrum[:, i] += tmp.sum(axis=(1, 2))
        spectrum /= 2*(nX*nY*nZ)**2

        energy_spectrum.append(spectrum)
        if verbose: print(" -- done !")

    return energy_spectrum

class OutputFiles():
    """
    Object to load and manipulate hdf5 Dedalus generated solution output
    """
    def __init__(self, folder, inference=False):
        self.folder = folder
        self.inference = inference
        fileNames = glob.glob(f"{self.folder}/*.h5")
        fileNames.sort(key=lambda f: int(f.split("_s")[-1].split(".h5")[0]))
        self.files = fileNames
        self._file = None   # temporary buffer to store the HDF5 file
        self._iFile = None  # index of the HDF5 stored in the temporary buffer
        vData0 = self.file(0)['tasks']['velocity']
        if self.inference:
             self.x = np.array(vData0[0,0,:,0])
             self.z = np.array(vData0[0,1,0,:])
             print(f'x-grid: {self.x.shape}')
             print(f'z-grid: {self.z.shape}')
             print(f'timesteps: {np.array(vData0[:,0,0,0]).shape}')
             self.dim = 2
        else:
            self.x = np.array(vData0.dims[2]["x"])
            self.dim = dim = len(vData0.dims)-2
            if dim == 2:
                self.z = np.array(vData0.dims[3]["z"])
                self.y = self.z
            elif dim == 3:
                self.y = np.array(vData0.dims[3]["y"])
                self.z = np.array(vData0.dims[4]["z"])
            else:
                raise NotImplementedError(f"{dim = }")


    def file(self, iFile:int):
        if iFile != self._iFile:
            try:
                self._file.close()
            except: pass
            self._iFile = iFile
            self._file = h5py.File(self.files[iFile], mode='r')
        return self._file

    def __del__(self):
        try:
            self._file.close()
        except: pass

    @property
    def nFiles(self):
        return len(self.files)

    @property
    def nX(self):
        return self.x.size

    @property
    def nY(self):
        return self.y.size

    @property
    def nZ(self):
        return self.z.size

    @property
    def shape(self):
        if self.dim == 2:
            return (4, self.nX, self.nZ)
        elif self.dim == 3:
            return (4, self.nX, self.nY, self.nZ)

    @property
    def k(self):
        if self.dim == 2:
            return getModes(self.x)
        elif self.dim == 3:
            return getModes(self.x), getModes(self.y)

    def vData(self, iFile:int):
        return self.file(iFile)['tasks']['velocity']

    def bData(self, iFile:int):
        return self.file(iFile)['tasks']['buoyancy']

    def pData(self, iFile:int):
        return self.file(iFile)['tasks']['pressure']

    def times(self, iFile:int=None):
        if iFile is None:
            return np.concatenate([self.times(i) for i in range(self.nFiles)])
        if self.inference:
            return np.array(self.vData(iFile)[:,0,0,0])
        else:
            return np.array(self.vData(iFile).dims[0]["sim_time"])

    @property
    def nFields(self):
        return [self.nTimes(i) for i in range(self.nFiles)]

    def fields(self, iField):
        offset = np.cumsum(self.nFields)
        iFile = np.argmax(iField < offset)
        iTime = iField - sum(offset[:iFile])
        data = self.file(iFile)["tasks"]
        fields = [
            data["velocity"][iTime, 0],
            data["velocity"][iTime, 1],
            ]
        if self.dim == 3:
            fields += [data["velocity"][iTime, 2]]
        fields += [
            data["buoyancy"][iTime],
            data["pressure"][iTime]
            ]
        return np.array(fields)

    def nTimes(self, iFile:int):
        return self.times(iFile).size

    def readField(self, iFile, name, iBeg=0, iEnd=None, step=1, verbose=False):
        if verbose: print(f"Reading {name} from hdf5 file {iFile}")
        if name == "velocity":
            fData = self.vData(iFile)
        elif name == "buoyancy":
            fData = self.bData(iFile)
        elif name == "pressure":
            fData = self.pData(iFile)
        else:
            raise ValueError(f"cannot read {name} from file")
        shape = fData.shape
        if iEnd is None: iEnd = shape[0]
        rData = range(iBeg, iEnd, step)
        data = np.zeros((len(rData), *shape[1:]))
        for i, iData in enumerate(rData):
            if verbose: print(f" -- field {i+1}/{len(rData)}, idx={iData}")
            data[i] = fData[iData]
        if verbose: print(" -- done !")
        return data

    def getMeanSpectrum(self, iFile:int, iBeg=0, iEnd=None, step=1, verbose=False, batchSize=5):
        """
        Mean spectrum from a given output file

        Parameters
        ----------
        iFile : int
            Index of the file to use.
        iBeg : int, optional
            Starting index for the fields to use. The default is 0.
        iEnd : int, optional
            Stopping index (non included) for the fields to use. The default is None.
        step : int, optional
            Index step for the fields to use. The default is 1.
        verbose : bool, optional
            Display infos message in stdout. The default is False.
        batchSize : int, optional
            Number of fields to regroup when computing one FFT. The default is 5.

        Returns
        -------
        spectra : np.ndarray[nT,size]
            The spectrum values for all nT fields.
        """
        spectra = []
        if iEnd is None:
            iEnd = self.nFields[iFile]
        subRanges = decomposeRange(iBeg, iEnd, step, batchSize)
        for iBegSub, iEndSub, stepSub in subRanges:
            if verbose:
                print(f" -- computing for fields in range ({iBegSub},{iEndSub},{stepSub})")
            velocity = self.readField(iFile, "velocity", iBegSub, iEndSub, stepSub, verbose)
            spectra += computeMeanSpectrum(velocity, verbose=verbose, xGrid=self.x, zGrid=self.z)
        return np.concatenate(spectra)

    def getFullMeanSpectrum(self, iBeg:int, iEnd=None):
        """
        Function to get full mean spectrum

        Args:
            iBeg (int): starting file index
            iEnd (int, optional): stopping file index. Defaults to None.

        Returns:
           sMean (np.ndarray): mean spectrum
           k (np.ndarray): wave number
        """
        if iEnd is None:
            iEnd = self.nFiles
        sMean = []
        for iFile in range(iBeg, iEnd):
            energy_spectrum = self.getMeanSpectrum(iFile)
            sx, sz = energy_spectrum                        # (1,time_index,k)
            sMean.append(np.mean((sx+sz)/2, axis=0))        # mean over time ---> (2, k)
        sMean = np.mean(sMean, axis=0)                      # mean over x and z ---> (k)
        np.savetxt(f'{self.folder}/spectrum.txt', np.vstack((sMean, self.k)))
        return sMean, self.k

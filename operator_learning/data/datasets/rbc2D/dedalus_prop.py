import h5py
import glob
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

def computeMeanSpectrum(uValues):
    """ uValues[nT, nVar, nX, nZ] """
    uValues = np.asarray(uValues)
    print(f"Computing Mean Spectrum on u[{', '.join([str(n) for n in uValues.shape])}]")

    energy_spectrum = []
    for i in range(2):
        u = uValues[:, i]                           # (nT, Nx, Nz)
        spectrum = np.fft.rfft(u, axis=-2)          # over Nx -->  #(nT, k, Nz)
        spectrum *= np.conj(spectrum)               # (nT, k, Nz)
        spectrum /= spectrum.shape[-2]              # normalize with Nx --> (nT, k, Nz)
        spectrum = np.mean(spectrum.real, axis=-1)  # mean over Nz --> (nT,k)
        energy_spectrum.append(spectrum)

    print(" -- done !")

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
        return (4, self.nX, self.nZ)
       

    @property
    def k(self):
        return getModes(self.x)
       

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
            spectra += computeMeanSpectrum(velocity)
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

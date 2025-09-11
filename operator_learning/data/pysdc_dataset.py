import glob
import h5py
import numpy as np
from typing import List, Optional, Union
from operator_learning.utils.misc import print_rank0
from pySDC.helpers.fieldsIO import FieldsIO

    
class PySDCReader:
    """
    Loader and accessor for pySDC data files for RBC3D simulations.

    Parameters
    ----------
    file_names : list of str
        List of paths to pySDC files.
    dim : int, default=3
        Dimensionality of the problem.
    """

    def __init__(self, file_names: List[str], dim: int = 3):
        self.dim = dim
        self.files = file_names

        # Extract grid info from first file
        first_file = self._open_file(0)
        self.x, self.y, self.z = first_file.header["coords"]
        print(
            f"x-grid: {self.x.shape}, y-grid: {self.y.shape}, z-grid: {self.z.shape}"
        )

    # ---------------------------
    # Internal utilities
    # ---------------------------
    def _open_file(self, idx: int):
        """Open a file by index."""
        return FieldsIO.fromFile(self.files[idx])

    def _read_range(
        self, iFile: int, field_idx: Union[int, List[int]], iBeg: int = 0, iEnd: Optional[int] = None
    ) -> np.ndarray:
        """General routine for reading fields over a range of timesteps."""
        field = self._open_file(iFile)   
        n_fields = field.nFields
        if iEnd is None:
            iEnd = n_fields
    
        if isinstance(field_idx, int):
            field_idx = [field_idx]
    
        data = []
        for i in range(iBeg, iEnd):
            _, arr = field.readField(i)  
            data.append(arr[field_idx, ...])
        return np.array(data)

    # ---------------------------
    # Properties
    # ---------------------------
    @property
    def nFiles(self) -> int:
        return len(self.files)

    @property
    def nX(self) -> int:
        return self.x.size

    @property
    def nY(self) -> int:
        return self.y.size

    @property
    def nZ(self) -> int:
        return self.z.size

    @property
    def fieldShape(self) -> tuple:
        return (5, self.nX, self.nY, self.nZ)

    @property
    def nFields(self) -> List[int]:
        """Number of fields per file (list)."""
        return [self._open_file(i).nFields for i in range(self.nFiles)]

    # ---------------------------
    # Data access
    # ---------------------------
    def vData(self, iFile: int, iBeg: int = 0, iEnd: Optional[int] = None) -> np.ndarray:
        """Velocity fields (u,v,w)."""
        return self._read_range(iFile, [0, 1, 2], iBeg, iEnd)

    def bData(self, iFile: int, iBeg: int = 0, iEnd: Optional[int] = None) -> np.ndarray:
        """Buoyancy field."""
        return self._read_range(iFile, 3, iBeg, iEnd)

    def pData(self, iFile: int, iBeg: int = 0, iEnd: Optional[int] = None) -> np.ndarray:
        """Pressure field."""
        return self._read_range(iFile, 4, iBeg, iEnd)

    def times(self, iFile: int) -> np.ndarray:
        """Return time values for given file."""
        return self._open_file(iFile).times

    def readSolution(
        self, iFile: int, iBeg: int = 0, iEnd: Optional[int] = None
    ) -> np.ndarray:
        """Return full solution array (u,v,w,T,p)."""
        return self._read_range(iFile, [0, 1, 2, 3, 4], iBeg, iEnd)

    def nTimes(self, iFile: int) -> int:
        """Number of time steps in file."""
        return len(self.times(iFile))

    def readField(
        self,
        iFile: int,
        name: str,
        iBeg: int = 0,
        iEnd: Optional[int] = None,
        step: int = 1,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Generic field reader.

        Parameters
        ----------
        iFile : int
            File index.
        name : {"velocity", "buoyancy", "pressure"}
            Field to read.
        iBeg : int, default=0
            Start index.
        iEnd : int, optional
            End index.
        step : int, default=1
            Sampling stride.
        verbose : bool, default=False
            Print progress.
        """
        field_map = {
            "velocity": self.vData,
            "buoyancy": self.bData,
            "pressure": self.pData,
        }
        if name not in field_map:
            raise ValueError(f"Unsupported field '{name}'")

        data = field_map[name](iFile, iBeg, iEnd)
        rData = range(0, data.shape[0], step)

        out = np.zeros((len(rData), *data.shape[1:]))
        for i, idx in enumerate(rData):
            if verbose:
                print(f" -- field {i+1}/{len(rData)}, idx={idx}")
            out[i] = data[idx]
        if verbose:
            print(" -- done !")
        return out

def createDatasetFromPySDC(
    dataDir: str,
    inSize: int,
    outStep: int,
    inStep: int,
    outType: str,
    outScaling: float,
    dataFile: str,
    verbose: bool = False,
    nDim: int = 3,
    **kwargs,
):
    """
    Create HDF5 dataset from PySDC simulation outputs.
    Supports simulations with different numbers of samples.
    """

    assert inSize == 1, "inSize != 1 not implemented yet ..."
    assert nDim == 3, "only 3D PySDC supported for now"

    # --- Gather simulations
    simFiles = glob.glob(f"{dataDir}/*.pySDC")
    nSimu = int(kwargs.get("nSimu", len(simFiles)))
    simFiles = simFiles[:nSimu]
    print_rank0("Using Simulations:")
    for s in simFiles:
        print_rank0(f" -- {s}")

    # --- Retrieve metadata from first simulation
    reader0 = PySDCReader([simFiles[0]])
    fieldShape = reader0.fieldShape
    times0 = reader0.times(0)
    xGrid, yGrid, zGrid = reader0.x, reader0.y, reader0.z

    dtData = times0[1] - times0[0]
    dtInput = dtData * outStep
    dtSample = dtData * inStep

    iBeg = int(kwargs.get("iBeg", 0))

    # --- compute ranges per simulation
    reader = PySDCReader(simFiles)
    simRanges = []
    nSamplesTotal = 0
    for iSim, file in enumerate(simFiles):
        nFields = reader.nFields[iSim]
        iEnd = int(kwargs.get("iEnd", nFields))
        sRange = range(iBeg, iEnd - inSize - outStep + 1, inStep)
        simRanges.append(sRange)
        nSamplesTotal += len(sRange)
        print_rank0(f"Simulation {iSim}: {len(sRange)} samples")

    # --- Write metadata to HDF5
    infoParams = {
        "inSize": inSize,
        "outStep": outStep,
        "inStep": inStep,
        "outType": outType,
        "outScaling": outScaling,
        "iBeg": iBeg,
        "dtData": dtData,
        "dtInput": dtInput,
        "xGrid": xGrid,
        "yGrid": yGrid,
        "zGrid": zGrid,
        "nSimu": nSimu,
        "nSamplesTotal": nSamplesTotal,
        "dtSample": dtSample,
    }

    print_rank0(f"Creating dataset from {nSimu} simulations, total {nSamplesTotal} samples ...")
    dataset = h5py.File(dataFile, "w")
    for name, val in infoParams.items():
        try:
            dataset.create_dataset(f"infos/{name}", data=np.asarray(val))
        except Exception:
            dataset.create_dataset(f"infos/{name}", data=val)

    dataShape = (nSamplesTotal, *fieldShape)
    print_rank0(f"data shape: {dataShape}")
    inputs = dataset.create_dataset("inputs", dataShape)
    outputs = dataset.create_dataset("outputs", dataShape)

    # --- Loop over simulations and samples
    sampleCounter = 0
    for iSim, file in enumerate(simFiles):
        sRange = simRanges[iSim]
        print_rank0(f" -- sampling {len(sRange)} samples from {file}")

        for iSample, iField in enumerate(sRange):
            if verbose:
                print_rank0(f"\t -- creating sample {iSample+1}/{len(sRange)}")

            inpt = reader.readSolution(iSim, iField, iField + 1)[0]
            outp = reader.readSolution(iSim, iField + outStep, iField + outStep + 1)[0]

            if outType == "update":
                outp = (outp - inpt) * outScaling

            inputs[sampleCounter] = inpt
            outputs[sampleCounter] = outp
            sampleCounter += 1

    dataset.close()
    print_rank0(" -- done !")

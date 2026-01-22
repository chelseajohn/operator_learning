import h5py
import numpy as np
import cupy as cp
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset
from operator_learning.utils.misc import print_rank0

class PICDataset(Dataset):

    def __init__(self, dataFile, **kwargs):
        """
        Dataset reader and getitem for PIC data

        Args:
            dataFile (hdf5): data file

        """

        self.dataFile = dataFile
        self._file = None
        self.dataClass = kwargs.get('dataClass', 'pic')

        if self.nDim == 2:
            self.kY = kwargs.get('kY', 12)
        else:
            self.kY = kwargs.get('kY', 12)
            self.kZ = kwargs.get('kZ', 12)

        self.kX = kwargs.get('kX', 12)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None  # remove the h5py file before pickling
        return state

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.dataFile, 'r')
        return self._file

    @property
    def inputs(self):
        return self.file['inputs']

    @property
    def outputs(self):
        return self.file['outputs']

    @property
    def nDim(self):
        return int(self.infos['nDim'][()])

    @property
    def outType(self):
        return self._decode(self.infos['outType'][()])

    @property
    def outScaling(self):
        return float(self.infos['outScaling'][()])

    def __len__(self):
        assert len(self.inputs) == len(self.outputs), \
            f"different sample number for inputs and outputs ({len(self.inputs)},{len(self.outputs)})"
        return len(self.inputs)

    def __getitem__(self, idx):
        inpt, outp = self.sample(idx)
        return torch.tensor(inpt), torch.tensor(outp)

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass

    def sample(self, idx):
        return self.inputs[idx], self.outputs[idx]

    @property
    def infos(self):
        return self.file["infos"]

    @property
    def output_prop(self):
        mean = self.infos['output_mean'][()]
        std = self.infos['output_std'][()]
        return mean, std

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

    def _decode(self, val):
        """Helper to decode HDF5 bytes into Python strings"""
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8")
        if isinstance(val, (list, tuple, np.ndarray)):
            return [ self._decode(v) for v in val ]
        return val

    def printInfos(self):
        print_rank0("### Dataset Infos ###")
        infos = self.infos
        print_rank0(f" -- nDim : {self.nDim}")
        print_rank0(f" -- inputKeys : {self._decode(infos['input_keys'][()])}")
        print_rank0(f" -- inputShape : {infos['input_shape'][()]}")
        print_rank0(f" -- outputKeys : {self._decode(infos['output_keys'][()])}")
        print_rank0(f" -- outputShape : {infos['output_shape'][()]}")
        print_rank0(f" -- outputMean : {infos['output_mean'][()]}")
        print_rank0(f" -- outputStd : {infos['output_std'][()]}")
        print_rank0(f" -- outType : {self._decode(infos['outType'][()])}")
        print_rank0(f" -- outScaling : {infos['outScaling'][()]:1.2g}")


def normalize_per_sample(data: cp.ndarray) -> cp.ndarray:
    """
    Normalize each sample independently to the [0, 1] range.
    """
    data_min = data.min(axis=1, keepdims=True)
    data_max = data.max(axis=1, keepdims=True)
    #denom = cp.where(data_max > data_min, data_max - data_min, 1.0)
    denom = np.where(data_max > data_min, data_max - data_min, 1.0)
    return (data - data_min) / denom


def normalize_global_zscore(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalize the entire dataset with global z-score normalization.
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std


def load_h5Dataset(file_path: str, keys: List[str], iEnd: Optional[int] = None, step: int = 1) -> List[np.ndarray]:
    """
    Load datasets from HDF5 file.

    Args:
        file_path: HDF5 file path
        keys: list of dataset keys
        iEnd: slicing end for weak Landau dataset
        step: subsampling stride

    Returns:
        List of numpy arrays
    """
    datasets = []
    with h5py.File(file_path, "r") as f:
        for key in keys:
            data = f[key][:(iEnd if key == 'pos_weakLandau' or key == 'Eout_weakLandau' else None):step, :]
            datasets.append(np.array(data, dtype=np.float32))
    return datasets


def createDatasetFromPIC(picFile: str,
                         dataFile: str,
                         iEnd: Optional[int] = None,
                         step: int = 1,
                         nDim: int = 1,
                         outType: str = 'solution',
                         outScaling: float = 1.0):
    # tested for 1D and 2D
    assert nDim in (1,2), 'tested only for 1D and 2D'
    input_keys = ["pos_weakLandau", "pos_strongLandau", "pos_tsi", "pos_bti"]
    output_keys = ["Eout_weakLandau", "Eout_strongLandau", "Eout_tsi", "Eout_bti"]

    # Load inputs and outputs
    inputs_list = load_h5Dataset(picFile, input_keys, iEnd, step)
    outputs_list = load_h5Dataset(picFile, output_keys, iEnd, step)


    inp = np.concatenate(inputs_list, axis=0) # 1D: (timestep, position), 2D: (timestep, position, dim)
    outp = np.concatenate(outputs_list, axis=0)  # 1D: (timstep, electricField), 2D: (timestep, electricField, dim)
    if nDim == 1:
        outputs = outp[:, np.newaxis, :]  # timestep, channel=1, electricField)
    else:
        inp = inp.swapaxes(-1,-2)  # (timstep, channel=dim, position)
        outputs = outp.swapaxes(-1,-2)   # (timstep, channel=dim, electricField)


    # Build Q array
    q1_xsize = sum(t.shape[0] for t in inputs_list[:3])
    q1_ysize = inputs_list[0].shape[1]
    q1 = np.full((q1_xsize, q1_ysize), -4 * np.pi, dtype=np.float32)
    q2 = np.full((inputs_list[-1].shape[0], inputs_list[-1].shape[1]), -2 * np.pi / 0.21, dtype=np.float32)
    Q = np.concatenate((q1, q2), axis=0)  # (timestep, position)
    # Stack Q as extra channel
    # shape: (timestep, dim+1, features); features: position, charge
    if nDim == 1:
        inputs = np.stack([inp, Q], axis=1)
    else:
        Q_new = Q[:, np.newaxis, :] # (timestep, 1, position)
        inputs = np.concatenate((inp, Q_new), axis=1)

    # Shuffle timestep
    perm = np.random.permutation(inputs.shape[0])
    inputs = inputs[perm]      # shape: (timestep, dim+1, features)
    outputs = outputs[perm]    # shape: (timestep, dim, field)

    # Normalize
    inputs[:, 0, :] = normalize_per_sample(inputs[:, 0, :])
    outputs[:, 0, :], meanEx, stdEx = normalize_global_zscore(outputs[:, 0, :])
    if nDim == 2:
         inputs[:, 1, :] = normalize_per_sample(inputs[:, 1, :])
         outputs[:, 1, :], meanEy, stdEy = normalize_global_zscore(outputs[:, 1, :])

    with h5py.File(dataFile, "w") as dataset:
        infoParams = {
            "nDim": nDim,
            "input_keys": input_keys,
            "output_keys": output_keys,
            "input_shape": inputs.shape,
            "output_shape": outputs.shape,
            "outType" : outType,
            "outScaling": outScaling,
        }
        if nDim == 1:
            infoParams.update({
                "output_mean": meanEx,
                "output_std": stdEx
                })
        else:
            infoParams.update({
                "output_mean": (meanEx, meanEy),
                "output_std": (stdEx, stdEy)
                })
        for name, val in infoParams.items():
            try:
                dataset.create_dataset(f"infos/{name}", data=np.asarray(val))
            except Exception:
                dataset.create_dataset(f"infos/{name}", data=val)

        # Datasets
        dataset.create_dataset("inputs", data=inputs)
        dataset.create_dataset("outputs", data=outputs)

    print(" -- done !")

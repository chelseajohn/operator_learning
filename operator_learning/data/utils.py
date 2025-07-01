import torch
from torch.utils.data import  DataLoader, random_split, Subset
from collections import defaultdict
from operator_learning.data.hdf5_dataset import HDF5Dataset, DomainDataset

def variable_tensor_collate_fn(batch):
    """
    Groups tensors of the same shape together and batches them separately.
    returns [nPatch_per_sample, trainSamples//nPatch_per_sample, 4, sx,sy]
    """
    grouped_tensors_inp = defaultdict(list)
    grouped_tensors_out = defaultdict(list)
    for element in batch:
        key = tuple(element[0].shape)               # input and output have same shape
        grouped_tensors_inp[key].append(element[0])
        grouped_tensors_out[key].append(element[0])
        
    batched_inp = [torch.stack(tensors) for tensors in grouped_tensors_inp.values()]  
    batched_out = [torch.stack(tensors) for tensors in grouped_tensors_out.values()]
    
    nBatched_lists = len(batched_inp)
    batchSize = [batched_inp[iBatch].shape[0] for iBatch in range(nBatched_lists)]

    # When using fixed domain sampling with add_fullGrid, make balanced batchSizes
    if len(set(batchSize)) > 1:
        min_batchSize = min(x for x in batchSize if x > 1)
        new_batched_inp  = []
        new_batched_out  = []
        for iBatch in range(nBatched_lists):
             split_inp = torch.split(batched_inp[iBatch], split_size_or_sections=min_batchSize, dim=0)
             split_out = torch.split(batched_out[iBatch], split_size_or_sections=min_batchSize, dim=0)
             new_batched_inp += split_inp
             new_batched_out += split_out
        # for i in range(len(new_batched_inp)):
        #     print(f'{i}: {new_batched_inp[i].shape}', flush=True)
        return (new_batched_inp, new_batched_out)

    # for i in range(len(batched_inp)):
    #     print(f'{i}: {batched_inp[i].shape}', flush=True)
    return (batched_inp, batched_out)  


def getDataLoaders(dataFile,
                   trainRatio=0.8, 
                   batchSize=20, seed=None, 
                   sampling_mode=None,
                   pad_to_fullGrid=False,
                   use_fixedPatch_startIdx=False,
                   nPatch_per_sample=1,
                   use_minLimit=False,
                   padding=[0,0,0,0], 
                   add_fullGrid=False, # to include full grid with domain grids
                   use_distributed_sampler=False,
                   **kwargs):

    if sampling_mode is not None:
        dataset = DomainDataset(dataFile,
                                sampling_mode,
                                pad_to_fullGrid, 
                                use_fixedPatch_startIdx,
                                nPatch_per_sample,
                                use_minLimit,
                                padding, 
                                **kwargs)
        
        if add_fullGrid:
            dataset.slices.append((dataset.nX, dataset.nY))
            dataset.nPatch_per_sample = len(dataset.slices)
            if use_fixedPatch_startIdx and not sampling_mode == 'ordered':
                dataset.patch_startIdx.append((0,0))
    else:
        dataset = HDF5Dataset(dataFile, **kwargs)


    dataset.printInfos()
    
    if (sampling_mode == 'random' and not pad_to_fullGrid) or \
       ((sampling_mode in ['fixed', 'ordered']) and (add_fullGrid or padding != [0,0,0,0])):
        nBatches = len(dataset)*dataset.nPatch_per_sample
        collate_fn = variable_tensor_collate_fn
        num_workers = 0
    else:
        nBatches = len(dataset)
        collate_fn = None
        num_workers = 4

    trainSize = int(trainRatio*nBatches)
    valSize = nBatches - trainSize

    if seed is None:
        trainIdx = list(range(0, trainSize))
        valIdx = list(range(trainSize, nBatches))
        trainSet = Subset(dataset, trainIdx)
        valSet = Subset(dataset, valIdx)
    else:
        generator = torch.Generator().manual_seed(seed)
        trainSet, valSet = random_split(
            dataset, [trainSize, valSize], generator=generator)

    if use_distributed_sampler:
        train_sampler = DistributedSampler(trainSet, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
        val_sampler = DistributedSampler(valSet, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
    else:
        train_sampler = None
        val_sampler = None


    if (sampling_mode == 'random' and not pad_to_fullGrid) or \
       ((sampling_mode in ['fixed', 'ordered']) and (add_fullGrid or padding != [0,0,0,0])):
        train_batchSize = len(trainSet)
        valid_batchSize = len(valSet)
    else:
        train_batchSize = batchSize
        valid_batchSize = batchSize

    trainLoader = DataLoader(trainSet, batch_size=train_batchSize, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    valLoader = DataLoader(valSet, batch_size=valid_batchSize, sampler=val_sampler, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return trainLoader, valLoader, dataset
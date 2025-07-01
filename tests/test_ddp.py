import os, sys
from pathlib import Path
base_path = Path(__file__).resolve().parents[1]
sys.path.append(str(base_path))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from operator_learning.utils.communication import Communicator
from operator_learning.model import FNO
from operator_learning.loss.data_loss import VectorNormLoss

def cleanup():
     dist.destroy_process_group()
     
def run_train(gpus_per_node, rank, model):
    print(f"Testing DDP training on rank {rank}")
    
    communicator = Communicator(gpus_per_node, rank)
    device = communicator.device
    
    model = FNO(**model).to(device)
    ddp_model = DDP(model, device_ids=[communicator.local_rank])
    
    loss_fn = VectorNormLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.00039)
    labels = torch.randn(5, 4, 256, 64).to(device)
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(5, 4, 256, 64))
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    cleanup()
    
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    model = {"da": 4, "du": 4, "dv": 16, "kX":12, "kY":12,
             "n_layers":4, "non_linearity": "gelu",
             "bias": False}
    if n_gpus < 2:
        print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    else:
        rank = int(os.getenv('RANK', '0'))
        run_train(n_gpus, rank, model)
  
      
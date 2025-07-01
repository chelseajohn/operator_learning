import os
import time
from pathlib import Path
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from operator_learning.data import getDataLoaders
from operator_learning.model import FNO
from operator_learning.loss import LOSSES_CLASSES
from operator_learning.utils.communication import Communicator
from operator_learning.utils.misc import print_rank0

class FourierNeuralOperator:
    
    TRAIN_DIR = None
    LOSSES_FILE = None
    USE_TENSORBOARD = True
    

    def __init__(self, data:dict=None, model:dict=None, optim:dict=None, 
                lr_scheduler:dict=None, parallel_strategy:dict=None,
                loss:dict=None, checkpoint=None, eval_only=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = int(os.getenv('RANK', '0'))
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))

        if parallel_strategy is not None:
            gpus_per_node = parallel_strategy.pop("gpus_per_node", 4)
            self.DDP_enabled = parallel_strategy.pop("ddp", False)
            if self.DDP_enabled: 
                self.communicator = Communicator(gpus_per_node, self.rank)
                self.world_size = self.communicator.world_size
                assert  self.world_size > 1, 'More than 1 GPU required for ditributed training'
                self.device = self.communicator.device
                self.rank = self.communicator.rank
                self.local_rank = self.communicator.local_rank
        else:
            self.DDP_enabled = False

        # Evaluation-only mode
        if eval_only:
            assert checkpoint is not None, "Checkpoint required for evaluation mode"
            if model is not None:
                self.modelConfig = model
            self.dataset = None
            self.load(checkpoint, modelOnly=True)
            return

        # Data loading
        assert "dataFile" in data, "Missing dataFile in data config"
        self.data_config = data.copy()
        self.xStep = data.pop("xStep", 1)
        self.yStep = data.pop("yStep", 1)
        self.zStep = data.pop("zStep", 1)
        data.pop("outType", None)
        data.pop("outScaling", None)
        self.use_domain_sampling = True if self.data_config['sampling_mode'] is not None else False

        # sample : [batchSize, 4, nX, nY, (nZ)]
        self.trainLoader, self.valLoader, self.dataset = getDataLoaders(**data,kX=model['kX'], kY=model['kY'], kZ=model['kZ'])
        self.outType = self.dataset.outType
        self.outScaling = self.dataset.outScaling

        # Loss
        if loss is None:    # Use default settings
            loss = {
                "name": "VectorNormLoss",
                "absolute": False,
            }
        assert "name" in loss, "Loss config must have a 'name'"
        loss_class = LOSSES_CLASSES.get(loss.pop("name"))
        if loss_class is None:
            raise NotImplementedError(f"Unknown loss type, available are {list(LOSSES_CLASSES.keys())}")

        if "grids" in loss:
            loss["grids"] = self.dataset.grid
        self.lossFunction = loss_class(**loss, device=self.device)
        self.lossConfig = loss

        # Loss tracking
        self.losses = {
            "model": {"avg_valid": -1, "avg_train": -1, "valid": -1, "train": -1},
            "id": {"valid": self.idLoss("valid"), "train": self.idLoss("train")},
        }
        
        print_rank0("### Model Infos ###")
        self.setupModel(model)
        self.setupOptimizer(optim)
        self.setupLRScheduler(lr_scheduler)

        if checkpoint is not None:
            self.load(checkpoint)
        else:
            self.epochs = 0
            
        self.tCompEpoch = 0
        self.gradientNormEpoch = 0.0
        self.writer = SummaryWriter(self.fullPath("tensorboard")) if self.USE_TENSORBOARD else None

    # -------------------------------------------------------------------------
    # Setup and utility methods
    # -------------------------------------------------------------------------
    def setupModel(self, model_config):
        self.model = FNO(**model_config).to(self.device)
        self.modelConfig = model_config.copy()
        print_rank0(self.modelConfig)
        model_df = self.model.print_size()
        print_rank0(model_df)
        if self.DDP_enabled:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        torch.cuda.empty_cache()

    def setupOptimizer(self, optim_config=None):
        optim_config = optim_config or {"name": "adam", "lr": 1e-4, "weight_decay": 1e-5}
        name = optim_config.pop("name")
        optim_class = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }.get(name)

        if optim_class is None:
            raise ValueError(f"Unknown optimizer: {name}")

        self.optimizer = optim_class(self.model.parameters(), **optim_config)
        self.optimConfig = optim_config
        self.optim = name

    def setupLRScheduler(self,lr_scheduler=None):
        if lr_scheduler is None:
            lr_scheduler = {"scheduler": "StepLR", "step_size": 100.0, "gamma": 0.98}
        self.scheduler_config = lr_scheduler
        scheduler = lr_scheduler.pop('scheduler')
        self.scheduler_name = scheduler
        if scheduler == "StepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **lr_scheduler)
        elif scheduler == "CosAnnealingLR":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **lr_scheduler)
        else:
            raise ValueError(f"LR scheduler {scheduler} not implemented yet")

    def idLoss(self, dataset_type="valid"):
        loader = self.valLoader if dataset_type == "valid" else self.trainLoader
        total_loss = 0.0
        nBatches = len(loader)
        data_iter = iter(loader)

        if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
            # [nBatches=nPatch_per_sample, batchSize=nSamples/nBatches, 4, nX, ny]
            inp_list, out_list = next(data_iter)  
            nBatches = len(inp_list)

        with torch.no_grad():
            for iBatch in range(nBatches):
                if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
                    inputs, outputs = (inp_list[iBatch], out_list[iBatch])
                else:
                    inputs, outputs = next(data_iter)
                if self.outType == "solution":
                    loss = self.lossFunction(inputs, outputs, inputs)
                elif self.outType == "update":
                    loss = self.lossFunction(torch.zeros_like(inputs), outputs, inputs)
                else:
                    raise ValueError(f"Invalid outType: {self.outType}")
                total_loss += loss.item()

        return total_loss / nBatches

    # -------------------------------------------------------------------------
    # Training methods
    # -------------------------------------------------------------------------
    def train(self):
        model = self.model.train()
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        nSamples = len(self.trainLoader.dataset)
        nBatches = len(self.trainLoader)
        batchSize = self.trainLoader.batch_size
        data_iter = iter(self.trainLoader)
        total_loss = 0.0
        gradsEpoch = 0.0
        idLoss = self.losses['id']['train']


        if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
            # [nBatches=nPatch_per_sample, batchSize=nSamples/nBatches, 4, nX, ny]
            inp_list, out_list = next(data_iter)
            nBatches = len(inp_list)
            batchSize = len(inp_list[0])

        for iBatch in range(nBatches):
            if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
                data = (inp_list[iBatch], out_list[iBatch])
            else:
                data = next(data_iter)
            inp = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
            ref = data[1][..., ::self.xStep, ::self.yStep].to(self.device)

            pred = model(inp)
            loss = self.lossFunction(pred, ref, inp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            grad_norm = grads.norm()
            gradsEpoch += grad_norm

            if self.USE_TENSORBOARD:
                self.writer.add_scalar("Gradients/Norm", grad_norm,iBatch)
                
            # print_rank0(f" At [{iBatch*batchSize + len(inp)}/{nSamples:>5d}] loss: {loss.item():>7f} (id: {idLoss:>7f}) -- lr: {optimizer.param_groups[0]['lr']}")
            total_loss += loss.item()

        if self.USE_TENSORBOARD:
            self.writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], self.epochs)

        scheduler.step()
        avg_loss = total_loss / nBatches

        if self.DDP_enabled:
            # Obtain the global average loss.
            ddp_loss = torch.Tensor([avg_loss]).to(self.device).clone()
            self.communicator.allreduce(ddp_loss,op=dist.ReduceOp.AVG)
            train_loss = ddp_loss.item()
        else:
            train_loss = avg_loss

        self.losses["model"]["train"] = train_loss
        self.gradientNormEpoch = gradsEpoch / nBatches
        print_rank0(f"Train Epoch {self.epochs}: Avg Loss={train_loss:.6f} (id: {idLoss:>7f}) -- lr: {optimizer.param_groups[0]['lr']}\n")

    def valid(self):
        model = self.model.eval()
        nBatches = len(self.valLoader)
        batchSize = self.valLoader.batch_size
        total_loss = 0.0
        idLoss = self.losses['id']['valid']
        data_iter = iter(self.valLoader)

        if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
            # [nBatches=nPatch_per_sample, batchSize=nSamples/nBatches, 4, nX, ny]
            inp_list, out_list = next(data_iter) 
            nBatches = len(inp_list)
            batchSize = len(inp_list[0])

        with torch.no_grad():
            for iBatch in range(nBatches):
                if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
                    data = (inp_list[iBatch], out_list[iBatch])
                else:
                    data = next(data_iter)
                inp = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
                ref = data[1][..., ::self.xStep, ::self.yStep].to(self.device)
                pred = model(inp)
                loss = self.lossFunction(pred,ref, inp)
                total_loss += loss.item()

        avg_loss = total_loss / nBatches
        if self.DDP_enabled:
            # Obtain the global average loss.
            ddp_loss = torch.Tensor([avg_loss]).to(self.device).clone()
            self.communicator.allreduce(ddp_loss,op=dist.ReduceOp.AVG)
            val_loss = ddp_loss.item()
        else:
            val_loss = avg_loss

        self.losses["model"]["valid"] = val_loss
        print_rank0(f"Validation Epoch {self.epochs}: Avg Loss={val_loss:.6f} (id: {idLoss:>7f})\n")

    def learn(self, nEpoch, save_interval=100):
        self.epochs += 1
        start_epoch = self.epochs
        end_epoch = start_epoch + nEpoch
        for i in range(start_epoch, end_epoch):
            print_rank0(f"\nEpoch {i}")
            t0_comp = time.perf_counter()
            self.train()
            self.valid()
            t_comp = time.perf_counter()- t0_comp
            self.tCompEpoch = t_comp

            t0_monit = time.perf_counter()
            self.monitor()
            t_monit = time.perf_counter()- t0_monit

            if i % save_interval == 0 or i == end_epoch-1 :
                t0_save = time.perf_counter()
                self.save(f'model_epoch{i}.pt')
                t_save = time.perf_counter() - t0_save
                print_rank0(f" --- End of epoch {self.epochs} (tComp: {t_comp:1.2e}s, tMonit: {t_monit:1.2e}s tSave: {t_save:1.2e}s) ---")

            self.epochs += 1
        
        print_rank0("Done Training!")

    def monitor(self):
        if self.USE_TENSORBOARD and self.rank == 0:
            self.writer.add_scalars("Losses", {
                "Train": self.losses["model"]["train"],
                "Valid": self.losses["model"]["valid"],
                "Train_id": self.losses["id"]["train"],
                "Valid_id": self.losses["id"]["valid"]
            }, self.epochs)
            self.writer.add_scalar("Gradients/NormEpoch", self.gradientNormEpoch, self.epochs)
            self.writer.flush()

        if self.LOSSES_FILE and self.rank == 0:
            with open(self.fullPath(self.LOSSES_FILE), "a") as f:
                f.write("{epochs}\t{train:1.18f}\t{valid:1.18f}\t{train_id:1.18f}\t{valid_id:1.18f}\t{gradNorm:1.18f}\t{tComp}\n".format(
                    epochs=self.epochs,
                    train_id=self.losses["id"]["train"], valid_id=self.losses["id"]["valid"],
                    gradNorm=self.gradientNormEpoch, tComp=self.tCompEpoch, **self.losses["model"]))
    
    def save(self, filename):
        path = self.fullPath(filename)
        checkpoint = {
            "model": self.modelConfig,
            "model_state_dict": self.model.state_dict(),
            "outType": self.outType,
            "outScaling": self.outScaling,
            "epochs": self.epochs,
            "losses": self.losses["model"],
            "optim": self.optim,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler_name,
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            }
        if self.rank == 0:
            torch.save(checkpoint, path)

    def load(self, filename, modelOnly=False):
        if self.DDP_enabled:
            map_location = {f'cuda:0': f'{self.device}'}
        else:
            map_location = self.device
        checkpoint = torch.load(self.fullPath(filename), map_location=self.device)

        if hasattr(self, "modelConfig") and self.modelConfig != checkpoint['model']:
            for key, value in self.modelConfig.items():
                if key not in checkpoint['model']:
                    checkpoint['model'][key] = value
            print_rank0("WARNING : different model settings in config file,"
                    " overwriting with config from checkpoint ...")
        print_rank0(f"Model: {checkpoint['model']}")
        self.setupModel(checkpoint["model"])
        state_dict = checkpoint['model_state_dict']
        # creating new OrderedDict for model trained without DDP but used now with DDP 
        # or model trained using DPP but used now without DDP
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if self.DDP_enabled:
                if k[:7] == 'module.':
                    name = k
                else:
                    name = 'module.'+ k
            else:
                if k[:7] == 'module.':
                    name = k[7:]
                else:
                    name = k    
            if v.dtype == torch.complex64:
                new_state_dict[name] = torch.view_as_real(v)
            else:
                new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.outType = checkpoint["outType"]
        self.outScaling = checkpoint["outScaling"]
        self.epochs = checkpoint.get("epochs")
        try:
            self.losses['model'] = checkpoint['losses']
        except AttributeError:
            self.losses = {"model": checkpoint['losses']}
        if not modelOnly:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # waiting for all ranks to load checkpoint
        if self.DDP_enabled:
            dist.barrier()

    @classmethod
    def fullPath(cls, path):
        if cls.TRAIN_DIR:
            os.makedirs(cls.TRAIN_DIR, exist_ok=True)
            return str(Path(cls.TRAIN_DIR) / path)
        return path


    # -------------------------------------------------------------------------
    # Inference method
    # -------------------------------------------------------------------------
    def __call__(self, u0, nEval=1):
        model = self.model.eval()
        inpt = torch.tensor(u0, device=self.device, dtype=torch.get_default_dtype())
    
        with torch.no_grad():
            for _ in range(nEval):
                outp = model(inpt)
                if self.outType == "update":
                    outp /= self.outScaling

                    # Mapping output to input shape to perform addition
                    if outp.shape == inpt.shape:
                        outp += inpt
                    else:
                        sliced_inpt = inpt[:,:,
                                      self.modelConfig['iXBeg']: self.modelConfig['iXEnd'],
                                      self.modelConfig['iYBeg']: self.modelConfig['iYEnd']]
                        # print_rank0(f'Sliced Input: {sliced_inpt.shape}')
                        outp += sliced_inpt
                inpt = outp

        u1 = outp.cpu().detach().numpy()
        return u1

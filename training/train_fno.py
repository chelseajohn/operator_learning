import os
import time
import numpy as np
from pathlib import Path
from collections import OrderedDict
from statistics import mean
import torch
import torch.distributed as dist
import cupy as cp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as tprof
import torch.cuda.nvtx as nvtx
from operator_learning.data import getDataLoaders
from operator_learning.model import FNO
from operator_learning.loss import LOSSES_CLASSES
from operator_learning.utils.communication import Communicator
from operator_learning.utils.misc import print_rank0, NoScale, compile_timing, optimizer_step, scheduler_step
from operator_learning.utils.misc import register_dtype_hooks, enable_tf32_only_on_a100


class FourierNeuralOperator:

    TRAIN_DIR = None
    LOSSES_FILE = 'loss.txt'
    USE_TENSORBOARD = True


    def __init__(self, data:dict=None, model:dict=None, optim:dict=None,
                lr_scheduler:dict=None, parallel_strategy:dict=None,
                loss:dict=None, profile:dict=None, checkpoint=None,
                eval_only=False, debug=False, device=None, benchmark=False, use_complex_amp=False,
                use_amp=False, compile=False, compile_mode='default', data_class='pic'):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.rank = int(os.getenv('RANK', '0'))
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.debug = debug
        self.benchmark = benchmark
        self.use_amp = use_amp
        self.use_complex_amp = use_complex_amp  # explicit casting to Float16 for complex numbers
        assert not (use_complex_amp and not use_amp), "use_complex_amp=True requires use_amp=True"

        self.compile = compile
        self.compile_mode = compile_mode

        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            self.autocast_device_type = "cuda" if "cuda" in self.device else "cpu"

        if use_amp:
            self.scaler = torch.amp.GradScaler(self.autocast_device_type, enabled=use_amp)
        else:
            self.scaler = NoScale()

        if profile is not None:
            self.enable_profile = profile['enableProfiler']
            self.profiler_type = profile['profiler']
            self.profiler_dir = profile['profileDir']
            if self.profiler_type == "torch":
                activities = [tprof.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(tprof.ProfilerActivity.CUDA)
                self.profiler = tprof.profile(
                    activities=activities,
                    schedule=tprof.schedule(skip_first=0, wait=0, warmup=1, active=2, repeat=1),
                    on_trace_ready=tprof.tensorboard_trace_handler(self.profiler_dir),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                    with_modules=True
                )
                print_rank0(f"[Profiler] Torch profiler enabled, results will be written to {self.profiler_dir}")
        else:
            self.enable_profile = False
            self.profiler_type = None
            self.profiler = None

        if parallel_strategy is not None:
            gpus_per_node = parallel_strategy.get("gpus_per_node", 4)
            self.DDP_enabled = parallel_strategy.get("ddp", False)
            # Tensor Parallel implemented only for PIC problem with DSE transform
            self.TP_enabled = parallel_strategy.get("tp", False)
            self.tp_size = parallel_strategy.get("tp_size", 2) if self.TP_enabled else 1
            self.device_mesh = None
            self.tp_mesh = None
            self.dp_mesh = None
            self.dp_size = 1
           
            if self.DDP_enabled or self.TP_enabled:
                self.communicator = Communicator(gpus_per_node, self.rank)
                self.world_size = self.communicator.world_size
                assert  self.world_size > 1, 'More than 1 GPU required for ditributed training'
                self.device = self.communicator.device
                self.rank = self.communicator.rank
                self.local_rank = self.communicator.local_rank

            if self.DDP_enabled and self.TP_enabled:
                self.dp_size = self.world_size // self.tp_size
            elif self.DDP_enabled:
                self.dp_size = self.world_size
                
            if self.TP_enabled:
                assert (
                        self.world_size % self.tp_size == 0
                    ), f"World size {self.world_size} needs to be divisible by TP size {self.tp_size}"
                if self.DDP_enabled:
                    mesh_shape = (self.dp_size, self.tp_size)
                    mesh_name = ("dp", "tp")
                else:
                    mesh_shape = (self.world_size,)
                    mesh_name = ("tp",)
            
                self.device_mesh = init_device_mesh(device_type=self.autocast_device_type,
                                                mesh_shape=mesh_shape,
                                                mesh_dim_names=mesh_name
                                                )
                self.tp_mesh = self.device_mesh["tp"]
                self.dp_mesh = self.device_mesh["dp"] if self.DDP_enabled else None
                self.tp_rank = (
                        self.tp_mesh.get_local_rank()
                        if self.DDP_enabled
                        else self.tp_mesh.get_rank()
                    )
                self.dp_rank = self.dp_mesh.get_local_rank() if self.DDP_enabled else None
                
            print_rank0(f'Using DDP with {self.dp_size} GPUs and Tensor Parallel with {self.tp_size} GPUs.')
        else:
            self.DDP_enabled = False
            self.TP_enabled = False
            self.device_mesh = None
            self.dp_mesh = None


        # Evaluation-only mode
        if eval_only:
            assert checkpoint is not None, "Checkpoint required for evaluation mode"
            if model is not None:
                self.modelConfig = model
            self.dataset = None
            self.dataClass = data_class
            self.load(checkpoint, modelOnly=True)
            return

        # Data loading
        assert "dataFile" in data, "Missing dataFile in data config"
        self.data_config = data.copy()
        self.xStep = self.data_config.pop("xStep", 1)
        self.yStep = self.data_config.pop("yStep", 1)
        self.zStep = self.data_config.pop("zStep", 1)
        self.data_config.pop("outType", 'solution')
        self.data_config.pop("outScaling", 1.0)
        self.use_domain_sampling = True if self.data_config['sampling_mode'] is not None else False  # only for RBC 2D
        self.dataClass = data['dataClass']

        # sample RBC: [batchSize, channel, nX, nY, (nZ)], sample PIC: [batchSize, channel, dim]
        self.trainLoader, self.valLoader, self.dataset, self.train_sampler, self.val_sampler = getDataLoaders(
                                                                        **self.data_config,
                                                                         kX=model['kX'], kY=model['kY'], 
                                                                         kZ=model['kZ'], dp_mesh=self.dp_mesh,
                                                                        )
        self.outType = self.dataset.outType
        self.outScaling = self.dataset.outScaling

        # Loss
        if loss is None:    # Use default settings
            loss = {
                "name": "VectorNormLoss",
                "absolute": False,
            }
        assert "name" in loss, "Loss config must have a 'name'"
        self.loss_config = loss.copy()
        loss_class = LOSSES_CLASSES.get(self.loss_config.pop("name"))
        if loss_class is None:
            raise NotImplementedError(f"Unknown loss type, available are {list(LOSSES_CLASSES.keys())}")

        self.lossFunction = loss_class(**self.loss_config, device=self.device)

        # Loss tracking
        if self.dataClass == 'rbc':
            self.losses = {
                "model": {"valid": -1, "train": -1},
                "id": {"valid": self.idLoss("valid"), "train": self.idLoss("train")},
            }
        else:
            self.losses = {
                "model": {"valid": -1, "train": -1}
            }

        print_rank0("### Model Infos ###")

        if checkpoint is not None:
            self.load(checkpoint)
        else:
            self.setupModel(model)
            self.setupOptimizer(optim)
            self.setupLRScheduler(lr_scheduler)
            self.epochs = 0

        self.tCompEpoch = 0
        self.gradientNormEpoch = 0.0
        self.writer = SummaryWriter(self.fullPath("tensorboard")) if self.USE_TENSORBOARD else None

    # -------------------------------------------------------------------------
    # Setup and utility methods
    # -------------------------------------------------------------------------
    def setupModel(self, model_config):
        mesh = self.tp_mesh if self.TP_enabled else None
        self.model = FNO(**model_config, dataset=self.dataset, dataClass=self.dataClass,\
                          use_complex_amp=self.use_complex_amp, device=self.device, device_mesh=mesh).to(self.device)

        # hooks = register_dtype_hooks(self.model)
        self.modelConfig = model_config.copy()
        print_rank0(self.modelConfig)
        model_df = self.model.print_size()
        print_rank0(model_df)
        if self.DDP_enabled:
            if self.TP_enabled:
                self.dp_group = self.dp_mesh.get_group()
            else:
                self.dp_group = None
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             process_group=self.dp_group,
                             broadcast_buffers=False)
        torch.cuda.empty_cache()

    def setupOptimizer(self, optim_config=None):
        self.optim_config = optim_config.copy() or {"name": "adam", "lr": 1e-4, "weight_decay": 1e-5}
        name = self.optim_config.pop("name")
        optim_class = {
            "adam": torch.optim.Adam,
            "adamW": torch.optim.AdamW,
        }.get(name)

        if optim_class is None:
            raise ValueError(f"Unknown optimizer: {name}")

        self.optimizer = optim_class(self.model.parameters(), **self.optim_config)
        self.optimConfig = optim_config
        self.optim = name

    def setupLRScheduler(self,lr_scheduler=None):
        if lr_scheduler is None:
            lr_scheduler = {"scheduler": "StepLR", "step_size": 100.0, "gamma": 0.98}
        self.scheduler_config = lr_scheduler.copy()
        scheduler = self.scheduler_config.pop('scheduler')
        self.scheduler_name = scheduler
        if scheduler == "StepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.scheduler_config)
        elif scheduler == "CosAnnealingLR":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.scheduler_config)
        else:
            raise ValueError(f"LR scheduler {scheduler} not implemented yet")

    def idLoss(self, dataset_type="valid"):  # Relevant only for RBC problem
        loader = self.valLoader if dataset_type == "valid" else self.trainLoader
        total_loss = 0.0
        nBatches = len(loader)
        data_iter = iter(loader)

        if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
            # [nBatches=nPatch_per_sample, batchSize=nSamples/nBatches, channel, nX, ny]
            inp_list, out_list = next(data_iter)
            nBatches = len(inp_list)

        with torch.no_grad():
            for iBatch in range(nBatches):
                if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
                    inputs, outputs = (inp_list[iBatch], out_list[iBatch])
                else:
                    inputs, outputs = next(data_iter)
                if self.outType == "solution":
                    loss = self.lossFunction(inputs, outputs)
                elif self.outType == "update":
                    loss = self.lossFunction(torch.zeros_like(inputs), outputs)
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

        if self.benchmark:
            fwd_peak_mem = []
            bwd_peak_mem = []
            fwd_reserv_mem = []
            bwd_reserv_mem = []

        # Epoch
        if self.enable_profile:
            if self.profiler_type == "torch":
                self.profiler.start()
            nvtx.range_push(f"TrainEpoch_{self.epochs}")

        nBatches = len(self.trainLoader)
        data_iter = iter(self.trainLoader)
        total_loss = 0.0
        gradsEpoch = 0.0
        if self.dataClass == 'rbc':
            idLoss = self.losses['id']['train']
        else:
            idLoss = 0.0    # not relevant

        if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:  # only for RBC2D
            # [nBatches=nPatch_per_sample, batchSize=nSamples/nBatches, channel, nX, ny]
            inp_list, out_list = next(data_iter)
            nBatches = len(inp_list)

        for iBatch in range(nBatches):
            # Batch
            # if self.enable_profile:
            #     nvtx.range_push(f"TrainBatch_{iBatch}")
            with torch.autocast(device_type=self.autocast_device_type, dtype=torch.float16, enabled=self.use_amp):
                if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
                    data = (inp_list[iBatch], out_list[iBatch])
                else:
                    data = next(data_iter)
                if self.dataClass == 'pic':
                    # sharding particles across tp ranks
                    if self.TP_enabled:
                        nParticles = data[0].shape[-1]
                        particles_per_tp = nParticles //self.tp_size
                        start = self.tp_rank * particles_per_tp
                        end = start + particles_per_tp
                    else:
                        start = 0
                        end = None

                    inp = data[0][:,:, start:end].to(self.device)
                    ref = data[1][:,:, start:end].to(self.device)
                else:
                    inp = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
                    ref = data[1][..., ::self.xStep, ::self.yStep].to(self.device)

                # Forward pass
                if self.enable_profile:
                    nvtx.range_push("forward")
                pred = model(inp)
                if iBatch == 0 and self.epochs == 1:
                    print_rank0(f'Shape of input/GPU: {inp.shape} and shape of ouput/GPU: {pred.shape}')
                if self.enable_profile:
                    nvtx.range_pop()   # end forward

                if self.enable_profile:
                    nvtx.range_push("loss")
                loss = self.lossFunction(pred, ref)
                if self.enable_profile:
                    nvtx.range_pop() # end loss

            if self.TP_enabled:
                # All-reduce for logging (detached, outside graph)
                tensor_loss = loss.detach().clone()
                dist.all_reduce(tensor_loss, op=dist.ReduceOp.AVG, group=self.tp_mesh.get_group())
                total_loss += tensor_loss
                # if iBatch % 10 == 0:
                #     print_rank0(f"Train Epoch {self.epochs}: tp_loss: {tensor_loss}\n")
            else:
                total_loss += loss.detach()    
                        
            optimizer.zero_grad()

            if self.benchmark and iBatch % 10 == 0:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                fwd_peak_mem.append(allocated)
                fwd_reserv_mem.append(reserved)

            # Backward
            if self.enable_profile:
                nvtx.range_push("backward")
            self.scaler.scale(loss).backward()
            if self.enable_profile:
                nvtx.range_pop()  # end backward

            # if self.TP_enabled:
            #     # allreduce tp gradients
            #     for param in model.parameters():
            #         if param.grad is not None:
            #             dist.all_reduce(param.grad, group=self.tp_mesh.get_group())
            #             param.grad /= self.tp_size
    
            if self.debug:
                any_grad_nonzero = False
                for name, p in self.model.named_parameters():
                    if p.grad is not None and p.grad.abs().sum() > 0:
                        any_grad_nonzero = True
                        break
                print_rank0(f"[DEBUG] Any nonzero gradients: {any_grad_nonzero}")
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        print_rank0(f"[DEBUG] {name} has no gradient")
                    else:
                        print_rank0(f"[DEBUG] {name} grad mean: {param.grad.mean().item():.6e}")

                real_model = self.model.module if self.DDP_enabled else self.model
                param_before = real_model.P.layers[0].weight.clone()

            # Optimizer
            if self.enable_profile:
                nvtx.range_push("optimizer_step")
            optimizer_step(self.scaler, optimizer)
            if self.enable_profile:
                nvtx.range_pop() # end optimizer

            if self.benchmark and iBatch % 10 == 0:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                bwd_peak_mem.append(allocated)
                bwd_reserv_mem.append(reserved)

            if self.enable_profile:
                if self.profiler_type == "torch":
                    self.profiler.step()
            #     nvtx.range_pop() # end batch

            if self.debug:
                param_after = real_model.P.layers[0].weight
                print_rank0(f"Param changed: {not torch.allclose(param_before, param_after)}")
                param_change = (param_before - param_after).norm().item()
                print_rank0(f"[DEBUG] Parameter change norm: {param_change:.6e}")
                # check if params update + optimizer states populated
                with torch.no_grad():
                    first_param = next(model.parameters())
                    print_rank0(f"[DEBUG][train] Param[0] value sample: {first_param.view(-1)[0].item():.6e}")
                opt_state = optimizer.state_dict()
                if opt_state["state"]:
                    first_state = next(iter(opt_state["state"].values()))
                    for k, v in first_state.items():
                        if isinstance(v, torch.Tensor):
                            print_rank0(f"[DEBUG][train] Optimizer state {k}: mean={v.float().mean().item():.6e}")
                else:
                    print_rank0("[DEBUG][train] Optimizer state is EMPTY after step()!")

            grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            grad_norm = grads.norm()
            gradsEpoch += grad_norm

            if self.USE_TENSORBOARD:
                self.writer.add_scalar("Gradients/Norm", grad_norm,iBatch)

            # print_rank0(f" At [{iBatch*batchSize + len(inp)}/{nSamples:>5d}] loss: {loss.item():>7f} (id: {idLoss:>7f}) -- lr: {optimizer.param_groups[0]['lr']}")
            
        if self.USE_TENSORBOARD:
            self.writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], self.epochs)

        scheduler_step(scheduler)
        avg_loss = total_loss / nBatches
        # avg_loss = total_loss / len(self.trainLoader.dataset)

        if self.DDP_enabled:
            if self.enable_profile:
                nvtx.range_push(f"TrainEpoch_{self.epochs}_DDPLoss")
            # Obtain the global average loss.
            dist.all_reduce(avg_loss, 
                            op=dist.ReduceOp.AVG, 
                            group=self.dp_group)
            if self.enable_profile:
                nvtx.range_pop()  # end ddploss
        
        train_loss = avg_loss.item()  # loss per gpu
        self.losses["model"]["train"] = train_loss
        self.gradientNormEpoch = gradsEpoch / nBatches
        print_rank0(f"Train Epoch {self.epochs}: AvgLoss={train_loss:.4e} (id: {idLoss:>7f}) -- lr: {optimizer.param_groups[0]['lr']}\n")

        if self.benchmark:
            print_rank0(f"CUDA Memory for Fwd Pass - Allocated: {mean(fwd_peak_mem):.2f} MB")
            print_rank0(f"CUDA Memory for Fwd Pass - Reserved: {mean(fwd_reserv_mem):.2f} MB")
            print_rank0(f"CUDA Memory for Bwd Pass - Allocated: {mean(bwd_peak_mem):.2f} MB")
            print_rank0(f"CUDA Memory for Bwd Pass - Reserved: {mean(bwd_reserv_mem):.2f} MB")
            print_rank0(f"Estimate Activations Memory: {mean(fwd_peak_mem)-mean(bwd_peak_mem):.2f} MB")

        if self.enable_profile:
            if self.profiler_type == "torch":
                self.profiler.stop()
            nvtx.range_pop() # end epoch

    def valid(self):
        model = self.model.eval()
        nBatches = len(self.valLoader)
        # batchSize = self.valLoader.batch_size
        total_loss = 0.0
        relative_error = 0.0
        #median_error = torch.zeros(len(self.valLoader.dataset))
        local_errors = torch.zeros(nBatches)
        data_iter = iter(self.valLoader)

        if self.dataClass == 'rbc':
           idLoss = self.losses['id']['valid']
        else:
            idLoss = 0.0 # not relevant

        if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:  # only for RBC2D
            # [nBatches=nPatch_per_sample, batchSize=nSamples/nBatches, channel, nX, ny]
            inp_list, out_list = next(data_iter)
            nBatches = len(inp_list)
            # batchSize = len(inp_list[0])

        with torch.no_grad():
            for iBatch in range(nBatches):
                # Batch
                # if self.enable_profile:
                #     nvtx.range_push(f"ValEpoch_{self.epochs}_Batch_{iBatch}")
                if self.enable_profile:
                    nvtx.range_push("forward+loss")

                if self.use_domain_sampling and not self.data_config['pad_to_fullGrid']:
                    data = (inp_list[iBatch], out_list[iBatch])
                else:
                    data = next(data_iter)
                if self.dataClass == 'pic':
                    # sharding particles across tp ranks
                    if self.TP_enabled:
                        nParticles = data[0].shape[-1]
                        particles_per_tp = nParticles //self.tp_size
                        start = self.tp_rank * particles_per_tp
                        end = start + particles_per_tp
                    else:
                        start = 0
                        end = None

                    inp = data[0][:,:, start:end].to(self.device)
                    ref = data[1][:,:, start:end].to(self.device)
                else:
                    inp = data[0][..., ::self.xStep, ::self.yStep].to(self.device)
                    ref = data[1][..., ::self.xStep, ::self.yStep].to(self.device)

                pred = model(inp)

                # if self.enable_profile:
                #     nvtx.range_push("valLoss")
                local_loss = self.lossFunction(pred,ref)
                error = torch.mean(torch.abs(ref.flatten(start_dim=1) - pred.flatten(start_dim=1)))/torch.mean(torch.abs(ref.flatten(start_dim=1))) * 100
                local_errors[iBatch] = error.detach()
                if self.TP_enabled:
                    # only for logging 
                    loss_tensor = local_loss.detach().clone()
                    dist.all_reduce(loss_tensor, 
                                    op=dist.ReduceOp.AVG,
                                    group=self.tp_mesh.get_group())
                    total_loss += loss_tensor

                    error_tensor = error.detach().clone()
                    dist.all_reduce(error_tensor,
                                    op=dist.ReduceOp.AVG,
                                    group=self.tp_mesh.get_group())
                    relative_error += error_tensor
                     # if iBatch % 10 == 0:
                     #      print_rank0(f"Train Epoch {self.epochs}: tp_loss: {loss_tensor} tp_error: {error_tensor}\n")
                else:
                    total_loss += local_loss.detach()
                    relative_error += error.detach()
                # if self.enable_profile:
                #     nvtx.range_pop() # end loss
                #     nvtx.range_pop() # end batch
                if self.enable_profile:
                    nvtx.range_pop() # end forward

        avg_loss = total_loss/nBatches
        relative_error = relative_error / nBatches
        if self.DDP_enabled:
            if self.enable_profile:
                nvtx.range_push(f"ValEpoch_{self.epochs}_DDPLoss")
            # Obtain the global average loss.
            dist.all_reduce(avg_loss, 
                            op=dist.ReduceOp.AVG, 
                            group=self.dp_group)
            dist.all_reduce(relative_error,
                            op=dist.ReduceOp.AVG, 
                            group=self.dp_group)
            local_errors = local_errors.to(self.device)
            out = torch.zeros(self.world_size * local_errors.numel(),
                              device=local_errors.device,
                              dtype=local_errors.dtype)
            dist.all_gather_into_tensor(out, local_errors)
            median_error = out.median().item()
            # median_error = 0.0
            if self.enable_profile:
                nvtx.range_pop() # end ddploss
        else:
            median_error = local_errors.median().item()
      
        val_loss = avg_loss.item() # loss per gpu
        val_error = relative_error.item()
        self.losses["model"]["valid"] = val_loss
        print_rank0(f"Validation Epoch {self.epochs}: AvgLoss={val_loss:.4e} TestError={val_error:.2f}% MedianTestError={median_error:.4f}% (id: {idLoss:>7f})\n")

    def learn(self, nEpoch, save_interval=100):
        self.epochs += 1
        start_epoch = self.epochs
        end_epoch = start_epoch + nEpoch

        # benchmark metrics
        if self.benchmark:
            epoch_time = []
            compute_time = []
            train_time = []
            monitor_time = []
            checkpoint_time = []
            compile_times = []
            mode_name = "Compiled" if self.compile else "Eager"

        # torch.compile
        if self.compile:
            print_rank0(f"Compiling training function with mode={self.compile_mode}...")
            try:
                train_fn = torch.compile(self.train, mode=self.compile_mode)
            except Exception as e:
                print_rank0(f"[WARN] torch.compile failed, falling back to eager mode: {e}")
                train_fn = self.train
        else:
            train_fn = self.train

        for i in range(start_epoch, end_epoch):
            print_rank0(f"\nEpoch {i}")

            if self.train_sampler is not None: 
                self.train_sampler.set_epoch(i)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(i)

            t0_epoch = time.perf_counter()
            # start profiling only from 3 iteration
            if i == 3 and self.enable_profile and self.profiler_type == "nsys":
                torch.cuda.cudart().cudaProfilerStart()

            t0_comp = time.perf_counter()
            if self.benchmark:
                _, compile_time = compile_timing(lambda: train_fn())
                compile_times.append(compile_time)
                print_rank0(f"{mode_name} train time (epoch {i}): {compile_time:.4f}s")
            else:
                train_fn()
            t_train = time.perf_counter() - t0_comp
            self.valid()
            t_comp = time.perf_counter() - t0_comp
            self.tCompEpoch = t_comp

            t0_monit = time.perf_counter()
            self.monitor()
            t_monit = time.perf_counter() - t0_monit

            if i % save_interval == 0 or i == end_epoch-1 :
                if self.enable_profile:
                    nvtx.range_push("checkpointing")

                t0_save = time.perf_counter()
                self.save(f'model_epoch{i}.pt')
                t_save = time.perf_counter() - t0_save

                if self.enable_profile:
                    nvtx.range_pop()  # end checkpoint

                if self.benchmark:
                    checkpoint_time.append(t_save)

                print_rank0(f" --- End of epoch {self.epochs} (tComp: {t_comp:1.2e}s, tMonit: {t_monit:1.2e}s tSave: {t_save:1.2e}s) ---")

            t_epoch = time.perf_counter() - t0_epoch

            if self.benchmark and i > 1:
                epoch_time.append(t_epoch)
                compute_time.append(t_comp)
                train_time.append(t_train)
                monitor_time.append(t_monit)

            self.epochs += 1

        print_rank0("Done Training!")
        if self.enable_profile and self.profiler_type == "nsys":
            torch.cuda.cudart().cudaProfilerStop()

        if self.benchmark and len(epoch_time) > 0:
            num_epochs = len(epoch_time)
            total_epoch_time = sum(epoch_time)
            total_train_time = sum(train_time)
            total_compute_time = sum(compute_time)
            total_monitor_time = sum(monitor_time)
            total_checkpoint_time = sum(checkpoint_time)
            total_samples = num_epochs * (len(self.trainLoader.dataset) + len(self.valLoader.dataset))
            total_train_samples = num_epochs * len(self.trainLoader.dataset)
            samples_per_sec_train = int(total_train_samples/total_train_time)
            samples_per_sec = int(total_samples/total_compute_time)
            compile_time_mean = mean(compile_times[1:]) # not including first epoch

            data = {
                "Metric": ["NumEpochs", "TotalEpochTime (s)",
                            "TotalMonitorTime (s)", "TotalCheckpointTime (s)",
                            "TotalComputeTime (s)","TotalTrainTime (s)",
                            "MeanCompileTime (s)", "TotalTrainTimesteps",
                            "TotalTimesteps", "TrainTimesteps/s",
                            "Timesteps/s"],
                "Value": [  round(num_epochs,0),
                            round(total_epoch_time, 3),
                            round(total_monitor_time, 3),
                            round(total_checkpoint_time, 3),
                            round(total_compute_time, 3),
                            round(total_train_time, 3),
                            round(compile_time_mean, 3),
                            round(total_train_samples, 0),
                            round(total_samples, 0),
                            round(samples_per_sec_train, 0),
                            round(samples_per_sec, 0)],
                }

            print_rank0("\n=== Benchmark Summary ===")
            for metric, value in zip(data["Metric"], data["Value"]):
                print_rank0(f"{metric}: {value}")
            print_rank0("==========================\n")

    def monitor(self):
        if self.USE_TENSORBOARD and self.rank == 0:
            self.writer.add_scalars("Losses", {
                "Train": self.losses["model"]["train"],
                "Valid": self.losses["model"]["valid"]
            }, self.epochs)
            if self.dataClass == 'rbc':
                self.writer.add_scalars('IdLoss',{
                    "Train_id": self.losses["id"]["train"],
                    "Valid_id": self.losses["id"]["valid"]
                }, self.epochs)
            self.writer.add_scalar("Gradients/NormEpoch", self.gradientNormEpoch, self.epochs)
            self.writer.flush()

        if self.LOSSES_FILE and self.rank == 0:
            with open(self.fullPath(self.LOSSES_FILE), "a") as f:
                line = "{epochs}\t{train:1.18f}\t{valid:1.18f}\t{gradNorm:1.18f}\t{tComp}\n"
                format_dict = {
                    "epochs": self.epochs,
                    "train": self.losses["model"]["train"],
                    "valid": self.losses["model"]["valid"],
                    "gradNorm": self.gradientNormEpoch,
                    "tComp": self.tCompEpoch
                }

                if self.dataClass == "rbc":
                    if self.epochs == 1:
                        f.write("Epochs\t\tTrainLoss\t\tValidLoss\t\tTrainIdLoss\t\tValidIdLoss\t\tGradNorm\t\tComputeTime\n")
                    line = "{epochs}\t{train:1.18f}\t{valid:1.18f}\t{train_id:1.18f}\t{valid_id:1.18f}\t{gradNorm:1.18f}\t{tComp}\n"
                    format_dict.update({
                        "train_id": self.losses["id"]["train"],
                        "valid_id": self.losses["id"]["valid"]
                    })
                else:
                    if self.epochs == 1:
                        f.write("Epochs\t\tTrainLoss\t\tValidLoss\t\tGradNorm\t\tComputeTime\n")

                f.write(line.format(**format_dict))

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

        if self.debug:
            #  verify optimizer state saved
            print_rank0(f"[DEBUG][save] Saving optimizer with {len(self.optimizer.state_dict()['state'])} entries")

        if self.rank == 0:
            torch.save(checkpoint, path)

    def load(self, filename, modelOnly=False):
        if self.DDP_enabled:
            map_location = {f'cuda:0': f'{self.device}'}
        else:
            map_location = self.device
        checkpoint = torch.load(self.fullPath(filename), map_location=map_location)

        if hasattr(self, "modelConfig") and self.modelConfig != checkpoint['model']:
            for key, value in self.modelConfig.items():
                if key not in checkpoint['model']:
                    checkpoint['model'][key] = value
            print_rank0("WARNING : different model settings in config file,"
                    " overwriting with config from checkpoint ...")

        print_rank0(f"Model: {checkpoint['model']}")
        state_dict = checkpoint['model_state_dict']

        # creating new OrderedDict for model trained without DDP but used now with DDP
        # or model trained using DPP but used now without DDP
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if self.DDP_enabled:
                name = k if k.startswith('module.') else 'module.' + k
            else:
                name = k[7:] if k.startswith('module.') else k
            if v.dtype == torch.complex64:
                new_state_dict[name] = torch.view_as_real(v)
            else:
                new_state_dict[name] = v

        self.setupModel(checkpoint['model'])
        self.model.load_state_dict(new_state_dict)
        self.outType = checkpoint["outType"]
        self.outScaling = checkpoint["outScaling"]
        self.epochs = checkpoint.get("epochs")

        try:
            self.losses['model'] = checkpoint['losses']
        except AttributeError:
            self.losses = {"model": checkpoint['losses']}

        if not modelOnly:
            self.setupOptimizer({"name": checkpoint['optim']})
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Move optimizer state tensors to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.setupLRScheduler({"scheduler": checkpoint['lr_scheduler']}.update(checkpoint['lr_scheduler_state_dict']))
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            if self.debug:
                # inspect optimizer state after load
                opt_state = self.optimizer.state_dict()
                print_rank0(f"[DEBUG][load] Loaded optimizer with {len(opt_state['state'])} entries")
                for k, v in opt_state["state"].items():
                    for name, t in v.items():
                        if isinstance(t, torch.Tensor):
                            print_rank0(f"[DEBUG][load] Param {k} - {name}: device={t.device}, mean={t.float().mean().item():.6e}")


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
        enable_tf32_only_on_a100()
        model = self.model.eval()
        inpt = torch.tensor(u0, device=self.device, dtype=torch.get_default_dtype())

        with torch.no_grad():
            for _ in range(nEval):
                outp = model(inpt)
                if self.outType == "update":
                    outp /= self.outScaling
                    outp += inpt
                inpt = outp

        # u1 = outp.cpu().detach().numpy()
        u1 = cp.from_dlpack(outp.detach())
        return u1

import hydra
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import get_dataloader
from loggers import TensorboardLogger
from meters import RunningMeter
from transformer import VQGANTransformer
from utils import set_seed

logging.basicConfig(filename=None, encoding='utf-8', level=logging.DEBUG)


class Trainer:
    def __init__(self, config: DictConfig):

        self.logger = None

        self.log_every = config["log_every"]

        self.output_dir = Path(config.output_dir)

        self.last_epoch = -1

        self.log_grads = config.log_grads
        
        self.device_id = config.get('device_id', 0)
        self.rank = config.get('rank', 0)
        self.ddp = config.ddp
        self.config = config
        self.world_size = 1

        if self.rank == 0:
            self.logger = TensorboardLogger()
        
        self.device = torch.device(f"{config.device}:{self.device_id}")
        self.vqgan_transformer = VQGANTransformer(config).to(self.device)

        if config.ddp:
            self.vqgan_transformer = DDP(self.vqgan_transformer, device_ids=[self.device_id], find_unused_parameters=True)
            self.world_size = dist.get_world_size()
        
        self.opt = None
        self._set_optimizer()

        if config.resume_from is not None:
            self._load_state(config.resume_from)

        self.train_dataloader = get_dataloader(config.data_path, config.img_size, config.batch_size, config.num_workers, config.ddp)
        self.epochs = config.epochs
        
        track = ["train/running/ce_loss", "train/epoch/ce_loss"]
        
        self.running_meters = {t: RunningMeter(window_size=self.log_every, ddp=self.ddp) for t in track if "running" in t}
        self.epoch_meters = {t: RunningMeter(window_size=len(self.train_dataloader) // self.world_size, ddp=self.ddp) for t in track if "epoch" in t}
    
    def _set_optimizer(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self._unwrap().transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_embed")
        
        named_parameters = self._unwrap().transformer.named_parameters()
        param_dict = {pn: p for pn, p in named_parameters}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        self.opt = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))

    def _unwrap(self):
        if self.ddp:
            return self.vqgan_transformer.module
        return self.vqgan_transformer

    def _load_state(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.last_epoch = ckpt["epoch"] if "epoch" in ckpt else 0

        self._unwrap().transformer.load_state_dict(ckpt["transformer"])

        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])

        logging.info(f"Resume training from {ckpt_path} from last epoch {self.last_epoch}")
    
    def _save_checkpoint(self, epoch):
        if not self.rank == 0:
            return
        
        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)

        torch.save({"transformer": self._unwrap().transformer.state_dict(),
                    "opt": self.opt.state_dict(),
                    "epoch": epoch},
                    os.path.join(self.output_dir / "checkpoints", f"vqgan_transformer_epoch_{epoch}.pt"))
        
    def train(self):
        steps_per_epoch = len(self.train_dataloader)
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, self.epochs):

            if self.ddp:
                self.train_dataloader.sampler.set_epoch(epoch)

            with tqdm(range(steps_per_epoch), disable=not self.rank == 0) as pbar:
                for i, batch in enumerate(self.train_dataloader):
                    global_step = epoch * steps_per_epoch + i

                    batch = batch.to(self.device)

                    logits, targets = self.vqgan_transformer(batch)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    
                    if self.log_grads:
                        for n, p in self._unwrap().transformer.named_parameters():
                            if p.grad is None:
                                print(n)
                            else:
                                m = self.running_meters.get(f"grad/{n}", RunningMeter(window_size=self.log_every, ddp=self.ddp))
                                m.update(p.grad.data.norm(2).cpu().item())
                                self.running_meters[f"grad/{n}"] = m

                    self.running_meters["train/running/ce_loss"].update(loss.detach().cpu().item())     
                    self.epoch_meters["train/epoch/ce_loss"].update(loss.detach().cpu().item())

                    if i % (self.log_every + 1) == 0:
                        with torch.no_grad():
                            img, rec, halh_sample, full_sample = self._unwrap().log_imgs(batch[0:1, ...])
                            img_log = torch.cat((img, rec, halh_sample, full_sample), dim=-1)
                            img_log = img_log.add(1).mul(0.5).detach().cpu()
                            
                            logs = {name: meter.compute() for name, meter in self.running_meters.items()}
                            logs["train/img"] = img_log
                            
                            if self.rank == 0:
                                self.logger.log(logs, global_step)

                    pbar.set_postfix(
                        EPOCH=epoch,
                        LOSS=np.round(self.running_meters["train/running/ce_loss"].compute(), 5)
                    )
                    pbar.update(1)

                logs = {name: meter.compute() for name, meter in self.epoch_meters.items()}
                
                if self.rank == 0:
                    self.logger.log(logs, global_step)

                    self._save_checkpoint(epoch)

                self.last_epoch += 1


@hydra.main(config_path="../configs/", config_name="vqgan_transformer_celeba.yaml", version_base="1.3")
def main(config: DictConfig):
    set_seed(3910574)

    if config.ddp:
        logging.info(f"Setting up DDP!")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()

        OmegaConf.set_struct(config, True)
        with open_dict(config):
            config.rank = rank
            config.device_id = device_id

    trainer = Trainer(config)

    logging.info(f"Start training!")
    trainer.train()
    logging.info(f"Done training!")

    if config.ddp:
        dist.destroy_process_group()
        logging.info(f"Cleaned up DDP!")


if __name__ == '__main__':
    main()

    # add color logging
    # bs = 64 
    # lr = 1.4e-4
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as tv_utils
from torchmetrics.aggregation import RunningMean

from tqdm import tqdm
import numpy as np
import os

from pathlib import Path

import hydra
from omegaconf import DictConfig

from dataloader import get_dataloader
from transformer import VQGANTransformer

from loggers import TensorboardLogger
import logging
logging.basicConfig(filename=None, encoding='utf-8', level=logging.DEBUG)


import random


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


set_seed(3910574)


class Trainer:
    def __init__(self, config: DictConfig):

        self.device = torch.device(config.device)

        self.logger = None

        self.log_every = config["log_every"]

        self.output_dir = Path(config.output_dir)

        self.last_epoch = -1

        self.logger = TensorboardLogger()

        self.vqgan_transformer = VQGANTransformer(config).to(self.device)
        
        self.opt = None
        self._set_optimizer()

        if config.resume_from is not None:
            self._load_state(config.resume_from)

        self.train_dataloader = get_dataloader(config.data_path, config.img_size, config.batch_size, config.num_workers, config.ddp)
        self.epochs = config.epochs
        
        track = ["train/running/ce_loss", "train/epoch/ce_loss"]
        
        self.running_meters = {t: RunningMean(window=self.log_every) for t in track if "running" in t}
        self.epoch_meters = {t: RunningMean(window=len(self.train_dataloader)) for t in track if "epoch" in t}
    
    def _set_optimizer(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.vqgan_transformer.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_embed")

        param_dict = {pn: p for pn, p in self.vqgan_transformer.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        self.opt = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))

    def _load_state(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.last_epoch = ckpt["epoch"] if "epoch" in ckpt else 0
        self.vqgan_transformer.transformer.load_state_dict(ckpt["transformer"])

        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])

        logging.info(f"Resume training from {ckpt_path} from last epoch {self.last_epoch}")
    
    def _save_checkpoint(self, epoch):
        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)
        torch.save({"transformer": self.vqgan_transformer.transformer.state_dict(),
                    "opt": self.opt.state_dict(),
                    "epoch": epoch},
                    os.path.join(self.output_dir / "checkpoints", f"vqgan_transformer_epoch_{epoch}.pt"))
        
    def train(self):
        steps_per_epoch = len(self.train_dataloader)
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, self.epochs):
            with tqdm(range(steps_per_epoch)) as pbar:
                for i, batch in zip(pbar, self.train_dataloader):
                    global_step = epoch * steps_per_epoch + i

                    batch = batch.to(self.device)

                    logits, targets = self.vqgan_transformer(batch)
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    self.running_meters["train/running/ce_loss"].update(loss.detach().cpu())     
                    self.epoch_meters["train/epoch/ce_loss"].update(loss.detach().cpu())

                    if i % self.log_every == 0:
                        with torch.no_grad():
                            img, rec, halh_sample, full_sample = self.vqgan_transformer.log_imgs(batch[0:1, ...])
                            img_log = torch.cat((img, rec, halh_sample, full_sample), dim=-1)
                            img_log = img_log.add(1).mul(0.5).detach().cpu()
                            
                            logs = {name: meter.compute().item() for name, meter in self.running_meters.items()}
                            logs["train/img"] = img_log
                            
                            self.logger.log(logs, global_step)
                            
                            # tv_utils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=3)

                    pbar.set_postfix(
                        EPOCH=epoch,
                        LOSS=np.round(self.running_meters["train/running/ce_loss"].compute().item(), 5)
                    )
                    pbar.update(0)

                logs = {name: meter.compute().item() for name, meter in self.epoch_meters.items()}
                self.logger.log(logs, global_step)

                self._save_checkpoint(epoch)

                self.last_epoch += 1


@hydra.main(config_path="../configs/", config_name="vqgan_transformer_celeba.yaml")
def main(config: DictConfig):
    print(config.root_dir, config.work_dir, config.output_dir, config.log_dir)
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

    # add color logging
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
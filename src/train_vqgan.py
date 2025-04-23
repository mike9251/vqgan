import hydra
import logging
import numpy as np
from omegaconf import DictConfig
import os
from pathlib import Path
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import get_dataloader
from discriminator import NLayerDiscriminator
from loggers import TensorboardLogger
from lpips import LPIPS
from meters import RunningMeter
from model_utils import init_weights
from vqgan import VQGAN

logging.basicConfig(filename=None, encoding='utf-8', level=logging.DEBUG)


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


def get_item(x: torch.Tensor) -> float:
    return x.detach().cpu().item()


class Trainer:
    def __init__(self, config: DictConfig):

        self.log_every = config["log_every"]

        self.output_dir = Path(config.output_dir)

        self.last_epoch = -1

        self.device_id = config.get('device_id', 0)
        self.rank = config.get('rank', 0)
        self.ddp = config.ddp
        self.config = config

        if self.rank == 0:
            self.logger = TensorboardLogger()
        
        self.device = torch.device(f"{config.device}:{self.device_id}")

        self.vqgan = VQGAN(config).to(self.device)
        self.disc = NLayerDiscriminator().to(self.device)
        self.disc.apply(init_weights)
        
        self.opt_vqgan = torch.optim.Adam(self.vqgan.parameters(),  lr=config.lr, betas=(config.beta1, config.beta2))
        self.opt_disc = torch.optim.Adam(self.disc.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

        if config.resume_from is not None:
            self._load_state(config.resume_from)

        self.perceptual_loss = LPIPS().eval().to(self.device)

        self.train_dataloader = get_dataloader(config.data_path, config.img_size, config.batch_size, config.num_workers, config.ddp)
        self.epochs = config.epochs
        self.disc_factor = config.disc_factor
        self.disc_start = config["disc_start"]
        self.perc_loss_factor = config.perc_loss_factor
        self.rec_loss_factor = config.rec_loss_factor
        
        track = ["train/running/q_loss", "train/running/perceptual_loss", "train/running/rec_loss", "train/running/disc_factor",
                "train/running/lambda", "train/running/g_loss", "train/running/vq_loss", "train/running/d_loss",
                "train/epoch/q_loss", "train/epoch/perceptual_loss", "train/epoch/rec_loss", "train/epoch/disc_factor",
                "train/epoch/lambda", "train/epoch/g_loss", "train/epoch/vq_loss", "train/epoch/d_loss"]
        
        self.running_meters = {t: RunningMeter(window_size=self.log_every, ddp=self.ddp) for t in track if "running" in t}
        self.epoch_meters = {t: RunningMeter(window_size=len(self.train_dataloader), ddp=self.ddp) for t in track if "epoch" in t}
    
    def _load_state(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.last_epoch = ckpt["epoch"] if "epoch" in ckpt else 0
        self.vqgan.load_state_dict(ckpt["vqgan"])
        self.disc.load_state_dict(ckpt["disc"])

        if "vqgan_opt" in ckpt:
            self.opt_vqgan.load_state_dict(ckpt["vqgan_opt"])
        if "disc_opt" in ckpt:
            self.opt_disc.load_state_dict(ckpt["disc_opt"])

        logging.info(f"Resume training from {ckpt_path} from last epoch {self.last_epoch}")
    
    def _save_checkpoint(self, epoch):
        if self.rank != 0:
            return
        
        os.makedirs(self.output_dir / "checkpoints", exist_ok=True)

        torch.save({"vqgan": self.vqgan.state_dict(),
                    "disc": self.disc.state_dict(),
                    "vqgan_opt": self.opt_vqgan.state_dict(),
                    "disc_opt": self.opt_disc.state_dict(),
                    "epoch": epoch},
                    os.path.join(self.output_dir / "checkpoints", f"vqgan_disc_epoch_{epoch}.pt"))
        
    def train(self):
        steps_per_epoch = len(self.train_dataloader)
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, self.epochs):
            with tqdm(range(steps_per_epoch), disable=not self.rank == 0) as pbar:
                for i, batch in enumerate(self.train_dataloader):
                    batch = batch.to(self.device)

                    global_step = epoch * steps_per_epoch + i

                    rec, _, q_loss = self.vqgan(batch)

                    # vqgan update
                    perceptual_loss = self.perceptual_loss(batch, rec).mean()
                    
                    rec_loss = torch.abs(batch - rec).mean()
                    
                    perc_rec_loss = self.perc_loss_factor * perceptual_loss + self.rec_loss_factor * rec_loss
                    
                    fake_logits = self.disc(rec)
                    disc_factor = self.vqgan.adopt_disc_weight(self.disc_factor, global_step, self.disc_start)
                    gen_loss = -torch.mean(fake_logits)

                    lamb = self.vqgan.calculate_lambda(perc_rec_loss, gen_loss)

                    vqgan_loss = perc_rec_loss + q_loss + disc_factor * lamb * gen_loss

                    self.opt_vqgan.zero_grad()
                    vqgan_loss.backward(retain_graph=True)
                    self.opt_vqgan.step()

                    # disc update
                    fake_logits = self.disc(rec.detach())
                    real_logits = self.disc(batch.detach())

                    d_loss_fake = torch.mean(F.relu(1 + fake_logits))
                    d_loss_real = torch.mean(F.relu(1 - real_logits))
                    d_loss = disc_factor * 0.5 * (d_loss_fake + d_loss_real)
                    self.opt_disc.zero_grad()
                    d_loss.backward()
                    self.opt_disc.step()

                    self.running_meters["train/running/q_loss"].update(get_item(q_loss))
                    self.running_meters["train/running/perceptual_loss"].update(get_item(perceptual_loss))
                    self.running_meters["train/running/rec_loss"].update(get_item(rec_loss))
                    self.running_meters["train/running/disc_factor"].update(disc_factor)
                    self.running_meters["train/running/lambda"].update(get_item(lamb))
                    self.running_meters["train/running/g_loss"].update(get_item(gen_loss))
                    self.running_meters["train/running/vq_loss"].update(get_item(vqgan_loss))
                    self.running_meters["train/running/d_loss"].update(get_item(d_loss))
                    
                    self.epoch_meters["train/epoch/q_loss"].update(get_item(q_loss))
                    self.epoch_meters["train/epoch/perceptual_loss"].update(get_item(perceptual_loss))
                    self.epoch_meters["train/epoch/rec_loss"].update(get_item(rec_loss))
                    self.epoch_meters["train/epoch/disc_factor"].update(disc_factor)
                    self.epoch_meters["train/epoch/lambda"].update(get_item(lamb))
                    self.epoch_meters["train/epoch/g_loss"].update(get_item(gen_loss))
                    self.epoch_meters["train/epoch/vq_loss"].update(get_item(vqgan_loss))
                    self.epoch_meters["train/epoch/d_loss"].update(get_item(d_loss))

                    if i % self.log_every == 0:
                        with torch.no_grad():
                            batch = batch.detach().cpu()
                            rec = rec.detach().cpu()
                            real_fake_images = torch.cat([batch[:3].add(1).mul(0.5), rec[:3].add(1).mul(0.5)], dim=2)
                            
                            logs = {name: meter.compute() for name, meter in self.running_meters.items()}
                            logs["train/img"] = real_fake_images
                            
                            if self.rank == 0:
                                self.logger.log(logs, global_step)

                    pbar.set_postfix(
                        EPOCH=epoch,
                        VQ_Loss=np.round(self.running_meters["train/running/vq_loss"].compute(), 5),
                        DISC_Loss=np.round(self.running_meters["train/running/d_loss"].compute(), 3)
                    )
                    pbar.update(1)

                logs = {name: meter.compute() for name, meter in self.epoch_meters.items()}

                if self.rank == 0:
                    self.logger.log(logs, global_step)
                    self._save_checkpoint(epoch)

                self.last_epoch += 1


@hydra.main(config_path="../configs/", config_name="vqgan_celeba.yaml")
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
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output


    """
    If d_loss stays ~ desc_factor and g_loss too (or gradually increases) it's a good sign. It means that D slowly learns to distinguish
    real images from fake, it gives lower score to fake images and slightly higher score to real images (keeping their sum approximately
    the same). At the same G catches up.

    15 epochs and I can already cherry-pick some good results!!!


    batch_size = 14
    lr = 0.000021
    desc_factor = 0.8
    desc_start = 2500 (after 1 epoch)

    python src/train_vqgan.py data_path=/workspace/data/celeba_hq_256 resume_from=checkpoints/vqgan_disc_epoch_19.pt root_dir=/workspace/code/vqgan-main
    """

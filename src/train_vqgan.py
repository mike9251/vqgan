import torch
import torch.nn.functional as F
from torchvision import utils as tv_utils
from torchmetrics.aggregation import RunningMean

import argparse
from tqdm import tqdm
import numpy as np
import os

from dataloader import get_dataloader
from discriminator import NLayerDiscriminator
from lpips import LPIPS
from vqgan import VQGAN
from model_utils import init_weights

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
    def __init__(self, config):
        self.prep_train_dir()

        self.device = torch.device(config["device"])

        self.logger = None

        self.log_every = config["log_every"]

        self.last_epoch = -1

        self.logger = TensorboardLogger()

        self.vqgan = VQGAN(config).to(self.device)
        self.disc = NLayerDiscriminator().to(self.device)
        self.disc.apply(init_weights)
        
        self.opt_vqgan = torch.optim.Adam(self.vqgan.parameters(),  lr=config["lr"], betas=(config["beta1"], config["beta2"]))
        self.opt_disc = torch.optim.Adam(self.disc.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))

        if config["resume_from"]:
            self._load_state(config["resume_from"])

        self.perceptual_loss = LPIPS().eval().to(self.device)


        self.train_dataloader = get_dataloader(config["data_path"], config["img_size"], config["batch_size"], config["num_workers"], config["ddp"])
        self.epochs = config["epochs"]
        self.disc_factor = config["disc_factor"]
        self.disc_start = config["disc_start"]
        self.perc_loss_factor = config["perc_loss_factor"]
        self.rec_loss_factor = config["rec_loss_factor"]
        
        track = ["train/running/perceptual_loss", "train/running/rec_loss", "train/running/disc_factor",
                "train/running/lambda", "train/running/g_loss", "train/running/vq_loss", "train/running/d_loss",
                "train/epoch/perceptual_loss", "train/epoch/rec_loss", "train/epoch/disc_factor",
                "train/epoch/lambda", "train/epoch/g_loss", "train/epoch/vq_loss", "train/epoch/d_loss"]
        
        self.running_meters = {t: RunningMean(window=self.log_every) for t in track if "running" in t}
        self.epoch_meters = {t: RunningMean(window=len(self.train_dataloader)) for t in track if "epoch" in t}
    
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

        
    @staticmethod
    def prep_train_dir():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        
    def train(self):
        steps_per_epoch = len(self.train_dataloader)
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, self.epochs):
            with tqdm(range(steps_per_epoch)) as pbar:
                for i, batch in zip(pbar, self.train_dataloader):
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

                    self.opt_vqgan.zero_grad(set_to_none=True)
                    vqgan_loss.backward(retain_graph=True)

                    self.opt_vqgan.step()

                    # disc update
                    fake_logits = self.disc(rec.detach())
                    real_logits = self.disc(batch.detach())

                    d_loss_fake = torch.mean(F.relu(1 + fake_logits))
                    d_loss_real = torch.mean(F.relu(1 - real_logits))
                    d_loss = disc_factor * 0.5 * (d_loss_fake + d_loss_real)
                    self.opt_disc.zero_grad(set_to_none=True)
                    d_loss.backward()
                    self.opt_disc.step()

                    # add q_loss
                    self.running_meters["train/running/perceptual_loss"].update(perceptual_loss.detach().cpu())
                    self.running_meters["train/running/rec_loss"].update(rec_loss.detach().cpu())
                    self.running_meters["train/running/disc_factor"].update(disc_factor)
                    self.running_meters["train/running/lambda"].update(lamb.detach().cpu())
                    self.running_meters["train/running/g_loss"].update(gen_loss.detach().cpu())
                    self.running_meters["train/running/vq_loss"].update(vqgan_loss.detach().cpu())
                    self.running_meters["train/running/d_loss"].update(d_loss.detach().cpu())

                    self.epoch_meters["train/epoch/perceptual_loss"].update(perceptual_loss.detach().cpu())
                    self.epoch_meters["train/epoch/rec_loss"].update(rec_loss.detach().cpu())
                    self.epoch_meters["train/epoch/disc_factor"].update(disc_factor)
                    self.epoch_meters["train/epoch/lambda"].update(lamb.detach().cpu())
                    self.epoch_meters["train/epoch/g_loss"].update(gen_loss.detach().cpu())
                    self.epoch_meters["train/epoch/vq_loss"].update(vqgan_loss.detach().cpu())
                    self.epoch_meters["train/epoch/d_loss"].update(d_loss.detach().cpu())

                    if i % self.log_every == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat([batch[:3].add(1).mul(0.5), rec[:3].add(1).mul(0.5)], dim=2)
                            
                            logs = {name: meter.compute().item() for name, meter in self.running_meters.items()}
                            logs["train/img"] = real_fake_images.detach().cpu()
                            
                            self.logger.log(logs, global_step)
                            
                            # tv_utils.save_image(real_fake_images, os.path.join("results", f"{epoch}_{i}.jpg"), nrow=3)

                    pbar.set_postfix(
                        EPOCH=epoch,
                        VQ_Loss=np.round(self.running_meters["train/running/vq_loss"].compute().item(), 5),
                        DISC_Loss=np.round(self.running_meters["train/running/d_loss"].compute().item(), 3)
                    )
                    pbar.update(0)

                logs = {name: meter.compute().item() for name, meter in self.epoch_meters.items()}
                self.logger.log(logs, global_step)
                
                torch.save({"vqgan": self.vqgan.state_dict(),
                            "disc": self.disc.state_dict(),
                            "vqgan_opt": self.opt_vqgan.state_dict(),
                            "disc_opt": self.opt_disc.state_dict(),
                            "epoch": epoch},
                            os.path.join("checkpoints", f"vqgan_disc_epoch_{epoch}.pt"))

                self.last_epoch += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--img-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--img-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--data-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="mps", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=3, help='Input batch size for training (default: 6)')
    parser.add_argument('--num-workers', type=int, default=0, help='Num threads to load data in background(default: 0)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=0.1, help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perc-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--ddp', type=bool, default=False, help='DDP training or not')
    parser.add_argument('--resume-from', type=str, default="", help='Path to the checkpoint to resume training from')
    parser.add_argument('--log-every', type=int, default=100, help='Number of train steps before a logging step')

    args = parser.parse_args()
    # args.data_path = "/Users/petrushkovm/Downloads/celeba_hq_256"


    # Start a new wandb run to track this script.
    # logger = wandb.init(
    #     # Set the wandb entity where your project will be logged (generally your team name).
    #     entity="soapbox92",
    #     # Set the wandb project where this run will be logged.
    #     project="VQGAN_Celeba",
    #     # Track hyperparameters and run metadata.
    #     config={
    #         "learning_rate": args.lr,
    #     "epochs": args.epochs,
    #     "batch_size": args.batch_size,
    #     },
    # )
    
    trainer = Trainer(vars(args))

    # trainer.logger = logger

    trainer.train()

    # logger.finish()

    # add color logging
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

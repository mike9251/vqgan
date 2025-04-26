import hydra
import numpy as np
from omegaconf import DictConfig
import cv2
from tqdm import tqdm

from transformer import VQGANTransformer
import os
import torch
from utils import set_seed


def tensor_to_img(x: torch.Tensor):
    x = x.add(1.0).mul(0.5)
    img = x.cpu().numpy()[0]
    img = img.transpose(1, 2, 0)
    return np.clip(255 * img, 0, 255).astype(np.uint8)

def save_img(img, save_dir, fname):
    cv2.imwrite(os.path.join(save_dir, f"{fname}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                
@hydra.main(config_path="../configs/", config_name="generate_images.yaml", version_base="1.3")
def main(config: DictConfig):
    if config.random_seed is not None:
        set_seed(config.random_seed)

    model = VQGANTransformer(config).to(config.device)
    model.eval()

    with tqdm(range(config.num_images)) as pbar:
        for i in range(config.num_images):
            x = model.generate(config.img_size)
            img = tensor_to_img(x)
            save_img(img, config.output_dir, f"{i}_{config.img_size}")
            pbar.update(1)


if __name__ == '__main__':
    main()
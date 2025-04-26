# VQ-GAN + Transformer on CelebA

This repository contains code to train a **VQ-GAN** model on the CelebA dataset, and then train a Transformer (GPT-like model) on the learned discrete latent space for image generation. **It supports DDP training**.

The project is inspired by the "**Taming Transformers for High-Resolution Image Synthesis**" paper, but streamlined to work with a single dataset (CelebA) for simplicity.

# Main Components

### 1. VQ-GAN Training

Trains a Vector Quantized Generative Adversarial Network (VQ-GAN) on CelebA.

Includes perceptual loss (LPIPS), adversarial loss, and commitment loss.

After training, you get a discrete codebook and a decoder to reconstruct images.

### 2. Transformer Training

Trains a Transformer to model sequences of VQ-GAN codebook indices.

At inference time, the Transformer generates latent codes, and the VQ-GAN decoder turns them into full images.


# Setup

## Clone repository
git clone <your-repo-url>
cd <repo-name>

## Install dependencies
pip install -r requirements.txt

## Download Celeba 256 dataset <kaggle>

# Training

### Step 1: Train VQ-GAN

`python src/train_vqgan.py data_path=<path-to-celeba-dataset> log_dir=<where-to-store-logs>`

I trained this model for ~9 hours (52 epochs) using 2x Nvidia 4090 GPUs.

`torchrun --standalone --nnodes --nproc_per_node=2 src/train_vqgan.py data_path=<path-to-celeba-dataset> log_dir=<where-to-store-logs> batch_size=24 num_workers=8 lr=4.2e-5 disc_start=630 ddp=True`

## Discriminator loss
![alt text](img/image.png)
## Generator loss
![alt text](img/image-1.png)
## VQ-VAE loss components
![alt text](img/image-2.png)
## VQ-GAN loss
![alt text](img/image-3.png)

### Step 2: Train Transformer

`python src/train_transformer.py data_path=<path-to-celeba-dataset> log_dir=<where-to-store-logs> vqgan_weights=<path-to-vqgan-checkpoint-from-step-1>`

I trained this model for ~9 hours (52 epochs) using 2x Nvidia 4090 GPUs.

`torchrun --standalone --nnodes --nproc_per_node=2 src/train_transformer.py data_path=<path-to-celeba-dataset> log_dir=<where-to-store-logs> batch_size=64 num_workers=8 lr=1.4e-4 ddp=True`

# Inference

After both models are trained:

`python scripts/generate_samples.py --vqgan_ckpt <path-to-vqgan> --transformer_ckpt <path-to-transformer>`

This script samples new images by generating discrete codes with the Transformer and decoding them with the VQ-GAN decoder.

Acknowledgments

Taming Transformers for High-Resolution Image Synthesis

CompVis/taming-transformers GitHub

Notes

This project is simplified for research/educational purposes and focuses only on the CelebA dataset.

Code structure is modular to allow extensions to other datasets or model tweaks.

TODOs
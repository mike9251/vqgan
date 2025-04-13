import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.codebook = Codebook(config)
        
        self.pre_quant_conv = nn.Conv2d(config["latent_dim"], config["latent_dim"], 1, 1, 0)
        self.post_quant_conv = nn.Conv2d(config["latent_dim"], config["latent_dim"], 1, 1, 0)

    def forward(self, x):
        z = self.encoder(x)
        pre_quant_z = self.pre_quant_conv(z)
        z_q, ind, q_loss = self.codebook(pre_quant_z)
        post_quant_z_q = self.post_quant_conv(z_q)
        y = self.decoder(post_quant_z_q)

        return y, ind, q_loss
    
    def encode(self, x):
        z = self.encoder(x)
        pre_quant_z = self.pre_quant_conv(z)
        return self.codebook(pre_quant_z)
    
    def decode(self, z_q):
        post_quant_z_q = self.post_quant_conv(z_q)
        return self.decoder(post_quant_z_q)
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer_weight = self.decoder.model[-1].weight
        perceptual_loss_grad = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grad = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lam = torch.norm(perceptual_loss_grad) / (torch.norm(gan_loss_grad) + 1e-4)
        lam = torch.clamp(lam, 0, 1e4).detach()
        return 0.8 * lam
    
    def adopt_disc_weight(self, disc_factor, i, threshold, value=0.0):
        if i < threshold:
            disc_factor = value
        return disc_factor
    
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    device = torch.device("mps")

    model = VQGAN({"img_channels": 3, "latent_dim": 128, "num_vectors": 100, "beta": 0.5})
    model = model.to(device)

    x = torch.rand((4, 3, 256, 256), device=device)

    y, ind, q_loss = model(x)
    print(y.shape, q_loss)

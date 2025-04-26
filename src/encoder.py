import torch
import torch.nn as nn
from helper import ResidualBlock, NonLocalBlock, DownBlock, GroupNorm, Swish


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = set([16])
        num_res_blocks = 2
        resolution = 256

        layers = [nn.Conv2d(3, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_ch))
            
            if i != len(channels) - 2:
                layers.append(DownBlock(channels[i + 1]))
                resolution //= 2

        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], config["latent_dim"], 3, 1, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    model = Encoder({"img_channels": 3, "latent_dim": 100})
    model = model.to(device)

    x = torch.rand((6, 3, 256, 256), device=device)
    
    y = model(x)

    print(y.shape, y.device)

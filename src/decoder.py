import torch
import torch.nn as nn

from blocks import GroupNorm, NonLocalBlock, ResidualBlock, Swish, UpBlock


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        channels = [512, 256, 256, 128, 128]
        attn_resolutions = {16}
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(config["latent_dim"], in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels),
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i != 0:
                layers.append(UpBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, 3, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    model = Decoder({"latent_dim": 512})
    model = model.to(device)

    x = torch.rand((6, 512, 16, 16), device=device)

    y = model(x)

    print(y.shape, y.device)

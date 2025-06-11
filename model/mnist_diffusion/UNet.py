import torch
import torch.nn as nn
from model.auxiliary_functions import SinusoidalPosEmb


class MnistUNet(nn.Module):
    def __init__(self, config, time_emb_dim=128):
        super().__init__()

        # Time embedding
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Project time embedding to bottleneck channel size
        self.time_to_bottleneck = nn.Linear(time_emb_dim, 256)

        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # skip connection uses concat
            nn.ReLU()
        )

        # Output layer
        self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        x = x.float()
        # Embed timestep
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.enc1(x)         # shape: (B, 64, 28, 28)
        x2 = self.enc2(x1)        # shape: (B, 128, 28, 28)

        # Bottleneck + time embedding
        x3 = self.bottleneck(x2)  # shape: (B, 256, 28, 28)
        t_bottleneck = self.time_to_bottleneck(t_emb).view(-1, 256, 1, 1)
        x3 = x3 + t_bottleneck    # broadcasting

        # Decoder with skip connections
        x4 = self.dec2(x3)                            # shape: (B, 128, 28, 28)
        x5 = self.dec1(torch.cat([x4, x1], dim=1))    # concat with x1 (64 channels)

        return self.output(x5)  # shape: (B, 1, 28, 28)

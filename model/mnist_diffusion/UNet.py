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
        self.condition_proj = nn.Linear(time_emb_dim, 256)

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

        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Output
        self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: (B, 1, 28, 28) input image
        t: (B,) timestep tensor
        """
        x, conds = x
        x = x.float()

        # Time embedding
        t_emb = self.time_embedding(t)            # (B, time_emb_dim)
        t_emb = self.time_mlp(t_emb)              # (B, time_emb_dim)
        emb_bottleneck = self.condition_proj(t_emb).view(-1, 256, 1, 1)  # (B, 256, 1, 1)

        # Encoder
        x1 = self.enc1(x)                         # (B, 64, 28, 28)
        x2 = self.enc2(x1)                        # (B, 128, 28, 28)

        # Bottleneck
        x3 = self.bottleneck(x2) + emb_bottleneck # (B, 256, 28, 28)

        # Decoder with skip connections
        x4 = self.dec2(torch.cat([x3, x2], dim=1))  # (B, 128, 28, 28)
        x5 = self.dec1(torch.cat([x4, x1], dim=1))  # (B, 64, 28, 28)

        return self.output(x5)                    # (B, 1, 28, 28)


# # --- U-Net Architecture with Label Conditioning --- Does not work good so far
# class MnistUNet(nn.Module):
#     def __init__(self, config=None, time_emb_dim=128, num_classes=10, label_emb_dim=128):
#         super().__init__()
#
#         # Time embedding
#         self.time_embedding = SinusoidalPosEmb(time_emb_dim)
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, time_emb_dim),
#             nn.ReLU()
#         )
#
#         # Label conditioning
#         self.label_emb = nn.Embedding(num_classes, label_emb_dim)
#         self.label_mlp = nn.Sequential(
#             nn.Linear(label_emb_dim, time_emb_dim),
#             nn.ReLU()
#         )
#
#         # Project (time + label) embedding to bottleneck channel count
#         self.condition_proj = nn.Linear(time_emb_dim, 256)
#
#         # Encoder
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#
#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#
#         # Decoder
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#
#         # Output
#         self.output = nn.Conv2d(64, 1, kernel_size=3, padding=1)
#
#     def forward(self, samples, t):
#         """
#         samples: tuple of (x, cond)
#             x: (B, 1, 28, 28)
#             cond: (B,) or None
#         t: (B,) timestep tensor
#         """
#         x, conds = samples
#         x = x.float()
#
#         # Embed timestep
#         t_emb = self.time_embedding(t)            # (B, time_emb_dim)
#         t_emb = self.time_mlp(t_emb)              # (B, time_emb_dim)
#
#         # Embed label
#         if conds is not None:
#             y_emb = self.label_emb(conds)         # (B, label_emb_dim)
#             y_emb = self.label_mlp(y_emb)         # (B, time_emb_dim)
#         else:
#             y_emb = torch.zeros_like(t_emb)
#
#         # Combine time and label embeddings
#         emb = t_emb + y_emb                       # (B, time_emb_dim)
#         emb_bottleneck = self.condition_proj(emb).view(-1, 256, 1, 1)  # (B, 256, 1, 1)
#
#         # Encoder
#         x1 = self.enc1(x)                         # (B, 64, 28, 28)
#         x2 = self.enc2(x1)                        # (B, 128, 28, 28)
#
#         # Bottleneck with conditioning
#         x3 = self.bottleneck(x2) + emb_bottleneck  # (B, 256, 28, 28)
#
#         # Decoder
#         x4 = self.dec2(x3)                        # (B, 128, 28, 28)
#         x5 = self.dec1(torch.cat([x4, x1], dim=1))  # (B, 64, 28, 28)
#
#         # Output
#         return self.output(x5)                    # (B, 1, 28, 28)
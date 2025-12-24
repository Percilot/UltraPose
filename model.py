import torch
import torch.nn as nn
from diffusers.models import UNet2DModel


class ConditionedUnet(nn.Module):
    def __init__(self, image_size=128):
        super().__init__()

        self.image_size = image_size

        self.cond_d_projector = nn.Sequential(
            nn.Linear(19, 1 * image_size * image_size),
            nn.ReLU()
        )

        self.cond_v_projector = nn.Sequential(
            nn.Linear(513, 1 * image_size * image_size),
            nn.ReLU()
        )

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=2 + 1 + 1,
            out_channels=2,
            layers_per_block=2,
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, image, t, cond_d, cond_v):
        B = image.size(0)
        cond_d_map = self.cond_d_projector(cond_d).view(B, 1, self.image_size, self.image_size)
        cond_v_map = self.cond_v_projector(cond_v).view(B, 1, self.image_size, self.image_size)
        net_input = torch.cat([image, cond_d_map, cond_v_map], dim=1)
        return self.model(net_input, t).sample
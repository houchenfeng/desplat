import math
from functools import reduce
from operator import mul
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gsplat.cuda._wrapper import spherical_harmonics
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components import MLP
from nerfstudio.fields.base_field import Field


def _get_fourier_features(xyz: Tensor, num_features=3):
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2
        ** torch.linspace(
            0, num_features - 1, num_features, dtype=xyz.dtype, device=xyz.device
        ),
        2,
    )
    offsets = torch.tensor(
        [0, 0.5 * math.pi] * num_features, dtype=xyz.dtype, device=xyz.device
    )
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    feat = torch.sin(feat).view(-1, reduce(mul, feat.shape[1:]))
    return feat


class EmbeddingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # sh_coeffs = 4**2
        self.feat_in = 3
        input_dim = config.appearance_embedding_dim
        if config.app_per_gauss:
            input_dim += 6 * self.config.appearance_n_fourier_freqs + self.feat_in

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.feat_in * 2),
        )

        self.bg_head = nn.Linear(self.feat_in * 2, 3)

    def forward(self, gembedding, aembedding, features_dc, viewdir=None):
        del viewdir  # Viewdirs interface is kept to be compatible with prev. version
        if self.config.app_per_gauss and gembedding is not None:
            inp = torch.cat((features_dc, gembedding, aembedding), dim=-1)
        else:
            inp = aembedding
        offset, mul = torch.split(
            self.mlp(inp) * 0.01, [self.feat_in, self.feat_in], dim=-1
        )
        return offset, mul

    def get_bg_color(self, aembedding):
        return self.bg_head(self.mlp(aembedding))


class BackgroundModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # sh_coeffs = 4**2
        self.feat_in = 3
        input_dim = config.appearance_embedding_dim
        if config.app_per_gauss:
            input_dim += 6 * self.config.appearance_n_fourier_freqs + self.feat_in

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.feat_in * 2),
        )

    def forward(self, gembedding, aembedding, features_dc, viewdir=None):
        del viewdir  # Viewdirs interface is kept to be compatible with prev. version

        if self.config.app_per_gauss_bg:
            inp = torch.cat((features_dc, gembedding, aembedding), dim=-1)
        else:
            inp = aembedding
        offset, mul = torch.split(
            self.mlp(inp) * 0.01, [self.feat_in, self.feat_in], dim=-1
        )
        return offset, mul

class BGField(Field):
    def __init__(
        self,
        appearance_embedding_dim: int,
        implementation: Literal["tcnn", "torch"] = "torch",
        sh_levels: int = 4,
        layer_width: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.sh_dim = (sh_levels + 1) ** 2

        self.encoder = MLP(
            in_dim=appearance_embedding_dim,
            num_layers=num_layers - 1,
            layer_width=layer_width,
            out_dim=layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation=implementation,
        )
        self.sh_base_head = nn.Linear(layer_width, 3)
        self.sh_rest_head = nn.Linear(layer_width, (self.sh_dim - 1) * 3)
        # zero initialization
        self.sh_rest_head.weight.data.zero_()
        self.sh_rest_head.bias.data.zero_()

    def get_background_rgb(
        self, ray_bundle: RayBundle, appearance_embedding=None, num_sh=4
    ) -> Tensor:
        """Predicts background colors at infinity."""
        cur_sh_dim = (num_sh + 1) ** 2
        directions = ray_bundle.directions.view(-1, 3)
        x = self.encoder(appearance_embedding).float()
        sh_base = self.sh_base_head(x)  # [batch, 3]
        sh_rest = self.sh_rest_head(x)[
            ..., : (cur_sh_dim - 1) * 3
        ]  # [batch, 3 * (num_sh - 1)]
        sh_coeffs = (
            torch.cat([sh_base, sh_rest], dim=-1)
            .view(-1, cur_sh_dim, 3)
            .repeat(directions.shape[0], 1, 1)
        )
        colors = spherical_harmonics(
            degrees_to_use=num_sh, dirs=directions, coeffs=sh_coeffs
        )

        return colors

    def get_sh_coeffs(self, appearance_embedding=None) -> Tensor:
        x = self.encoder(appearance_embedding)
        base_color = self.sh_base_head(x)
        sh_rest = self.sh_rest_head(x)
        sh_coeffs = torch.cat([base_color, sh_rest], dim=-1).view(-1, self.sh_dim, 3)
        return sh_coeffs
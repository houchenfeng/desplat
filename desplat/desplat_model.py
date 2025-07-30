from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch._dynamo
from pytorch_msssim import SSIM
from torch import nn
from torch.nn import functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from desplat.field import (
    BGField,
    EmbeddingModel,
    _get_fourier_features,
)
from gsplat.cuda._wrapper import spherical_harmonics
from gsplat.rendering import rasterization
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.camera_utils import normalize
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import (
    RGB2SH,
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
    num_sh_bases,
    random_quat_tensor,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


def quat_to_rotmat(quat):
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


@dataclass
class DeSplatModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: DeSplatModel)

    ### Settings for DeSplat
    num_dynamic_points: int = 1000
    """Initial number of dynamic points"""
    refine_every_dyn: int = 10
    """period of steps where gaussians are culled and densified"""
    use_adc: bool = True
    """Whether to use ADC for dynamic points management"""
    enable_reset_alpha: bool = False
    """Whether to reset alpha for dynamic points"""
    continue_split_dyn: bool = False
    """Whether to continue splitting for dynamic points after step for splitting stops"""
    distance: float = 0.02
    """Distance of dynamic points from camera"""
    split_screen_size :float = 0.05
    """Screen size threshold for splitting dynamic points"""

    ### Settings for Regularization
    alpha_bg_loss_lambda: float = 0.01
    """Lambda for alpha background loss"""
    alpha_2d_loss_lambda: float = 0.01
    """Lambda for alpha loss of 2D dynamic Gaussians"""
    enable_bg_model: bool = False
    """Whether to enable background model"""
    bg_sh_degree: int = 4
    """Degree of SH bases for background model"""
    bg_num_layers: int = 3
    """Number of layers in the background model"""
    bg_layer_width: int = 128
    """Width of each layer in the background model"""

    ### Settings for Appearance Optimization
    enable_appearance: bool = False
    """Enable or disable appearance optimization"""
    app_per_gauss: bool = False
    """Whether to optimize appearance according to the per-Gaussian embedding"""
    appearance_embedding_dim: int = 32
    """Dimension of the appearance embedding"""
    appearance_n_fourier_freqs: int = 4
    """Number of Fourier frequencies for per-Gaussian embedding initialization"""
    appearance_init_fourier: bool = True
    """Whether to initialize the per-Gaussian embedding with Fourier frequencies"""

class DeSplatModel(SplatfactoModel):
    config: DeSplatModelConfig

    def populate_modules(self):
        cameras = self.kwargs["cameras"]
        self.dataparser_config = self.kwargs["dataparser"]

        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter(
                (torch.rand((self.config.num_random, 3)) - 0.5)
                * self.config.random_scale
            )

        assert cameras is not None

        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]

        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        # appearance embedding for each Gaussian
        embeddings = _get_fourier_features(
            means, num_features=self.config.appearance_n_fourier_freqs
        )
        embeddings.add_(torch.randn_like(embeddings) * 0.0001)
        if not self.config.appearance_init_fourier:
            embeddings.normal_(0, 0.01)

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        if self.config.app_per_gauss:
            # appearance embedding for each Gaussian
            embeddings = _get_fourier_features(means, num_features=self.config.appearance_n_fourier_freqs)
            embeddings.add_(torch.randn_like(embeddings) * 0.0001)
            if not self.config.appearance_init_fourier:
                embeddings.normal_(0, 0.01)
            self.gauss_params["embeddings"] = embeddings

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        self.camera_idx = 0
        self.camera = None

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        # only the storage method is dict we can use ADC
        self.populate_modules_dyn_dict()
        self.num_points_dyn = self.num_points_dyn_dict

        self.loss_threshold = 1.0
        self.max_loss = 0.0
        self.min_loss = 1e10

        # Add the appearance embedding
        if self.config.enable_appearance:
            self.appearance_embeddings = torch.nn.Embedding(
                self.num_train_data, self.config.appearance_embedding_dim
            )
            self.appearance_embeddings.weight.data.normal_(0, 0.01)
            self.appearance_mlp = EmbeddingModel(self.config)
        else:
            self.appearance_embeddings = None
            self.appearance_mlp = None

        if self.config.enable_bg_model:
            self.bg_model = BGField(
                appearance_embedding_dim=self.config.appearance_embedding_dim,
                implementation="torch",
                sh_levels=self.config.bg_sh_degree,
                num_layers=self.config.bg_num_layers,
                layer_width=self.config.bg_layer_width,
            )
        else:
            self.bg_model = None

    def populate_modules_dyn_dict(self):
        cameras = self.kwargs["cameras"]
        self.gauss_params_dyn_dict = nn.ModuleDict()
        num_points_dyn = self.config.num_dynamic_points

        self.optimizers_dyn = {}

        for i in range(self.num_train_data):
            camera = cameras[i]

            optimized_camera_to_world = camera.camera_to_worlds

            camera_rotation = optimized_camera_to_world[:3, :3]
            camera_position = optimized_camera_to_world[:3, 3]
            distance_to_cam = self.config.distance
            cube = torch.rand(num_points_dyn, 3)
            cube[:, 0] = (cube[:, 0] - 0.5) * 0.02
            cube[:, 1] = (cube[:, 1] - 0.5) * 0.02
            cube[:, 2] = distance_to_cam

            means_dyn = torch.nn.Parameter(
                camera_position.repeat(num_points_dyn, 1) - cube @ camera_rotation.T
            )

            distances_dyn, _ = self.k_nearest_sklearn(means_dyn.data, 3)
            distances_dyn = torch.from_numpy(distances_dyn)
            avg_dist_dyn = distances_dyn.mean(dim=-1, keepdim=True)
            scales_dyn = torch.nn.Parameter(torch.log(avg_dist_dyn.repeat(1, 3)))
            rgbs_dyn = torch.nn.Parameter(torch.rand(num_points_dyn, 3))
            quats_dyn = nn.Parameter(random_quat_tensor(num_points_dyn))
            opacities_dyn = nn.Parameter(
                torch.logit(0.1 * torch.ones(num_points_dyn, 1))
            )

            self.gauss_params_dyn_dict[str(i)] = nn.ParameterDict(
                {
                    "means_dyn": means_dyn,
                    "scales_dyn": scales_dyn,
                    "rgbs_dyn": rgbs_dyn,
                    "quats_dyn": quats_dyn,
                    "opacities_dyn": opacities_dyn,
                }
            )

        self.xys_dyn = {}
        self.radii_dyn = {}
        self.xys_grad_norm_dyn = {str(i): None for i in range(self.num_train_data)}
        self.max_2Dsize_dyn = {str(i): None for i in range(self.num_train_data)}
        self.vis_counts_dyn = {str(i): None for i in range(self.num_train_data)}

    @property
    def embeddings(self):
        if self.config.app_per_gauss:
            return self.gauss_params["embeddings"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in [
                "means",
                "scales",
                "quats",
                "features_dc",
                "features_rest",
                "opacities",
            ]:
                dict[f"gauss_params.{p}"] = dict[p]
            if self.config.app_per_gauss:
                dict[f"gauss_params.embeddings"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]

        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(
                torch.zeros(new_shape, device=self.device)
            )

        for i in range(self.num_train_data):
            if "means_dyn" in dict:
                for p in [
                    "means_dyn",
                    "scales_dyn",
                    "quats_dyn",
                    "rgbs_dyn",
                    "opacities_dyn",
                ]:  # "features_dc_dyn", "features_rest_dyn",
                    dict[f"gauss_params_dyn_dict.{str(i)}.{p}"] = dict[p]
            newp_dyn = dict[f"gauss_params_dyn_dict.{str(i)}.means_dyn"].shape[0]
            for name, param in self.gauss_params_dyn_dict[str(i)].items():
                old_shape = param.shape
                new_shape = (newp_dyn,) + old_shape[1:]
                self.gauss_params_dyn_dict[str(i)][name] = torch.nn.Parameter(
                    torch.zeros(new_shape, device=self.device)
                )

        super().load_state_dict(dict, **kwargs)

    def num_points_dyn_dict(self, i):
        return self.gauss_params_dyn_dict[str(i)]["means_dyn"].shape[0]

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval
                > self.num_train_data + self.config.refine_every
            )

            if do_densification:
                # then we densify
                # for static points
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )
                avg_grad_norm = (
                    (self.xys_grad_norm / self.vis_counts)
                    * 0.5
                    * max(self.last_size[0], self.last_size[1])
                )
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (
                    self.scales.exp().max(dim=-1).values
                    > self.config.densify_size_thresh
                ).squeeze()
                splits &= high_grads
                if self.step < self.config.stop_screen_size_at:
                    splits |= (
                        self.max_2Dsize > self.config.split_screen_size
                    ).squeeze()
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (
                    self.scales.exp().max(dim=-1).values
                    <= self.config.densify_size_thresh
                ).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)

                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat(
                            [param.detach(), split_params[name], dup_params[name]],
                            dim=0,
                        )
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)

            elif (
                self.step >= self.config.stop_split_at
                and self.config.continue_cull_post_densification
            ):
                deleted_mask = self.cull_gaussians()

            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)
            if (
                self.step < self.config.stop_split_at
                and self.step % reset_interval == self.config.refine_every
            ):
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(
                        torch.tensor(reset_value, device=self.device)
                    ).item(),
                )

                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

            if self.config.use_adc:
                do_densification_dyn = (
                    self.step < self.config.stop_split_at
                    or self.config.continue_split_dyn
                ) and self.step % (
                    self.num_train_data * self.config.refine_every_dyn
                ) < self.config.refine_every
                # for dynamic points
                if do_densification_dyn:
                    for camera_idx in range(self.num_train_data):
                        # if number of Gaussians == 0, skip the densification
                        if self.num_points_dyn(camera_idx) == 0:
                            self.xys_grad_norm_dyn[str(camera_idx)] = None
                            self.vis_counts_dyn[str(camera_idx)] = None
                            self.max_2Dsize_dyn[str(camera_idx)] = None
                            continue

                        # we densify the 2D image when every image has been seen for refine_every_dyn times
                        # For the dynamic points, for every image, we densify the points
                        xys_grad_norm_dyn = self.xys_grad_norm_dyn[str(camera_idx)]
                        vis_counts_dyn = self.vis_counts_dyn[str(camera_idx)]
                        max_2Dsize_dyn = self.max_2Dsize_dyn[str(camera_idx)]

                        dyn_gaussians = self.gauss_params_dyn_dict[str(camera_idx)]

                        assert (
                            xys_grad_norm_dyn is not None
                            and vis_counts_dyn is not None
                            and max_2Dsize_dyn is not None
                        )
                        avg_grad_norm_dyn = (
                            (xys_grad_norm_dyn / vis_counts_dyn)
                            * 0.5
                            * max(self.last_size[0], self.last_size[1])
                        )
                        high_grads_dyn = (
                            avg_grad_norm_dyn > self.config.densify_grad_thresh
                        ).squeeze()
                        splits_dyn = (
                            dyn_gaussians["scales_dyn"].exp().max(dim=-1).values
                            > self.config.densify_size_thresh
                        ).squeeze()
                        splits_dyn &= high_grads_dyn
                        if self.step < self.config.stop_screen_size_at:
                            splits_dyn |= (
                                max_2Dsize_dyn > self.config.split_screen_size
                            ).squeeze()
                        nsamps = self.config.n_split_samples

                        split_params_dyn = self.split_gaussians_dyn(
                            camera_idx, splits_dyn, nsamps
                        )

                        dups_dyn = (
                            dyn_gaussians["scales_dyn"].exp().max(dim=-1).values
                            <= self.config.densify_size_thresh
                        ).squeeze()
                        dups_dyn &= high_grads_dyn
                        dup_params_dyn = self.dup_gaussians_dyn(camera_idx, dups_dyn)

                        for name, param in self.gauss_params_dyn_dict[
                            str(camera_idx)
                        ].items():
                            self.gauss_params_dyn_dict[str(camera_idx)][name] = (
                                torch.nn.Parameter(
                                    torch.cat(
                                        [
                                            param.detach(),
                                            split_params_dyn[name],
                                            dup_params_dyn[name],
                                        ],
                                        dim=0,
                                    )
                                )
                            )

                        # append zeros to the max_2Dsize tensor
                        self.max_2Dsize_dyn[str(camera_idx)] = torch.cat(
                            [
                                self.max_2Dsize_dyn[str(camera_idx)],
                                torch.zeros_like(split_params_dyn["scales_dyn"][:, 0]),
                                torch.zeros_like(dup_params_dyn["scales_dyn"][:, 0]),
                            ],
                            dim=0,
                        )

                        split_idcs = torch.where(splits_dyn)[0]
                        # log this info to logfile

                        self.dup_in_all_optim_dyn(
                            camera_idx, optimizers, split_idcs, nsamps
                        )
                        dup_idcs = torch.where(dups_dyn)[0]
                        self.dup_in_all_optim_dyn(camera_idx, optimizers, dup_idcs, 1)

                        # After a guassian is split into two new gaussians, the original one should also be pruned.
                        splits_mask_dyn = torch.cat(
                            (
                                splits_dyn,
                                torch.zeros(
                                    nsamps * splits_dyn.sum() + dups_dyn.sum(),
                                    device=self.device,
                                    dtype=torch.bool,
                                ),
                            )
                        )

                        vis_cull_mask = (self.radii_dyn[str(camera_idx)] < 0.01).squeeze()  # cull invisible distractor Gaussians
                        splits_mask_dyn = splits_mask_dyn | vis_cull_mask

                        deleted_mask_dyn = self.cull_gaussians_dyn(
                            camera_idx, splits_mask_dyn
                        )

                        if deleted_mask_dyn is not None:
                            self.remove_from_all_optim_dyn(
                                camera_idx, optimizers, deleted_mask_dyn
                            )

                        self.xys_grad_norm_dyn[str(camera_idx)] = None
                        self.vis_counts_dyn[str(camera_idx)] = None
                        self.max_2Dsize_dyn[str(camera_idx)] = None

                elif (
                    self.step >= self.config.stop_split_at
                    and self.config.continue_cull_post_densification
                    and self.step % (
                        self.num_train_data * self.config.refine_every_dyn
                    ) < self.config.refine_every
                ):
                    for camera_idx in range(self.num_train_data):
                        if self.num_points_dyn(camera_idx) == 0:
                            self.xys_grad_norm_dyn[str(camera_idx)] = None
                            self.vis_counts_dyn[str(camera_idx)] = None
                            self.max_2Dsize_dyn[str(camera_idx)] = None
                            continue
                        
                        deleted_mask_dyn = self.cull_gaussians_dyn(camera_idx)
                        if deleted_mask_dyn is not None:
                            self.remove_from_all_optim_dyn(
                                camera_idx, optimizers, deleted_mask_dyn
                            )
                        self.xys_grad_norm_dyn[str(camera_idx)] = None
                        self.vis_counts_dyn[str(camera_idx)] = None
                        self.max_2Dsize_dyn[str(camera_idx)] = None


                reset_dyn = (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval == self.config.refine_every
                    and self.config.enable_reset_alpha
                )
                # reset
                if reset_dyn:
                    # Reset value for dynamic points
                    for camera_idx in range(self.num_train_data):
                        reset_value = self.config.cull_alpha_thresh * 2.0
                        dyn_gaussians = self.gauss_params_dyn_dict[str(camera_idx)]

                        dyn_gaussians["opacities_dyn"].data = torch.clamp(
                            dyn_gaussians["opacities_dyn"],
                            max=torch.logit(
                                torch.tensor(reset_value, device=self.device)
                            ).item(),
                        )

                        # reset the exp of optimizer
                        optim = optimizers.optimizers["opacities_dyn"]

                        param = optim.param_groups[0]["params"][camera_idx]
                        param_state = optim.state[param]
                        if "exp_avg" in param_state:
                            param_state["exp_avg"] = torch.zeros_like(
                                param_state["exp_avg"]
                            )
                            param_state["exp_avg_sq"] = torch.zeros_like(
                                param_state["exp_avg_sq"]
                            )

    def cull_gaussians_dyn(self, i, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points_dyn(i)
        # cull transparent ones
        culls = (
            torch.sigmoid(self.gauss_params_dyn_dict[str(i)]["opacities_dyn"])
            < self.config.cull_alpha_thresh
        ).squeeze()  # self.config.cull_alpha_thresh
        # if the point is invisible for all the camera, cull it

        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None and extra_cull_mask.shape[0] == n_bef:
            culls = culls | extra_cull_mask

        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (
                torch.exp(self.gauss_params_dyn_dict[str(i)]["scales_dyn"])
                .max(dim=-1)
                .values
                > self.config.cull_scale_thresh
            ).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                if self.max_2Dsize_dyn[str(i)] is not None:
                    toobigs = (
                        toobigs
                        | (
                            self.max_2Dsize_dyn[str(i)] > self.config.cull_screen_size
                        ).squeeze()
                    )
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params_dyn_dict[str(i)].items():
            self.gauss_params_dyn_dict[str(i)][name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Dynamic Culled {n_bef - self.num_points_dyn(i)} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points_dyn(i)} remaining)"
        )

        return culls

    def split_gaussians_dyn(self, i, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(
            f"Dynamic Splitting {split_mask.sum().item()/self.num_points_dyn(i)} gaussians: {n_splits}/{self.num_points_dyn(i)}"
        )
        centered_samples = torch.randn(
            (samps * n_splits, 3), device=self.device
        )  # Nx3 of axis-aligned scales

        # if split_mask.sum().item() == 0:
        #     out = {}
            
        #     for name, param in self.gauss_params_dyn_dict[str(i)].items():
        #         if name not in out:
        #             out[name] = param.repeat(samps, 1)
        #     return out

        # print("self.gauss_params_dyn_dict[str(i)]['scales_dyn']",self.gauss_params_dyn_dict[str(i)]['scales_dyn'].shape)
        # print("self.gauss_params_dyn_dict[str(i)]['scales_dyn'][split_mask]",str(i),self.gauss_params_dyn_dict[str(i)]["scales_dyn"][split_mask].shape)
        # print("split_mask", split_mask.shape,split_mask)
        # mask=self.gauss_params_dyn_dict[str(i)]["scales_dyn"][split_mask]
        # print( mask)
        if len(split_mask.shape) == 0:
            split_mask = torch.tensor([True], device='cuda:0')
        # print("split_mask",split_mask.shape)
        # print("samps",samps)
        # print("self.gauss_params_dyn_dict[str(i)]['scales_dyn'][split_mask]",str(i),self.gauss_params_dyn_dict[str(i)]["scales_dyn"][split_mask].shape)

        # print("self.gauss_params_dyn_dict[str(i)]['scales_dyn']",self.gauss_params_dyn_dict[str(i)]['scales_dyn'].shape)
        scaled_samples = (
            torch.exp(
                self.gauss_params_dyn_dict[str(i)]['scales_dyn'][split_mask].repeat(
                    samps, 1
                )
            )
            * centered_samples
        )  # how these scales are rotated
        
        # print('mask',mask.shape,mask)
        quats = self.gauss_params_dyn_dict[str(i)]["quats_dyn"][
            split_mask
        ] / self.gauss_params_dyn_dict[str(i)]["quats_dyn"][split_mask].norm(
            dim=-1, keepdim=True
        )  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.gauss_params_dyn_dict[str(i)]["means_dyn"][
            split_mask
        ].repeat(samps, 1)
        # step 2, sample new colors
        new_colors = self.gauss_params_dyn_dict[str(i)]["rgbs_dyn"][split_mask].repeat(
            samps, 1
        )
        # new_features_dc = self.gauss_params_dyn_dict[str(i)]["features_dc_dyn"][split_mask].repeat(samps, 1)
        # new_features_rest = self.gauss_params_dyn_dict[str(i)]["features_rest_dyn"][split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.gauss_params_dyn_dict[str(i)]["opacities_dyn"][
            split_mask
        ].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(
            torch.exp(self.gauss_params_dyn_dict[str(i)]["scales_dyn"][split_mask])
            / size_fac
        ).repeat(samps, 1)
        self.gauss_params_dyn_dict[str(i)]["scales_dyn"][split_mask] = torch.log(
            torch.exp(self.gauss_params_dyn_dict[str(i)]["scales_dyn"][split_mask])
            / size_fac
        )
        # step 5, sample new quats
        new_quats = self.gauss_params_dyn_dict[str(i)]["quats_dyn"][split_mask].repeat(
            samps, 1
        )

        out = {
            "means_dyn": new_means,
            "rgbs_dyn": new_colors,
            "opacities_dyn": new_opacities,
            "scales_dyn": new_scales,
            "quats_dyn": new_quats,
        }
        for name, param in self.gauss_params_dyn_dict[str(i)].items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians_dyn(self, i, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(
            f"Dynamic Duplication: Duplicating {dup_mask.sum().item()/self.num_points_dyn(i)} gaussians: {n_dups}/{self.num_points_dyn(i)}"
        )
        new_dups = {}
        for name, param in self.gauss_params_dyn_dict[str(i)].items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def dup_in_all_optim_dyn(self, i, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups_dyn_dict()

        for group, _ in param_groups.items():
            param = param_groups[group][i]
            self.dup_in_optim_dyn(
                i, optimizers.optimizers[group], dup_mask, param, n
            )

        self.radii_dyn[str(i)] = torch.cat(
            [self.radii_dyn[str(i)], self.radii_dyn[str(i)][dup_mask.squeeze()].repeat(n,)], dim=0
        )

    def dup_in_optim_dyn(self, i, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][i][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(
                1 for _ in range(param_state["exp_avg"].dim() - 1)
            )
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(
                        *repeat_dims
                    ),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(
                        param_state["exp_avg_sq"][dup_mask.squeeze()]
                    ).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params] = param_state
        optimizer.param_groups[0]["params"][i] = new_params
        del param

    def remove_from_all_optim_dyn(self, i, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups_dyn_dict()
        for group, _ in param_groups.items():
            param = param_groups[group][i]
            self.remove_from_optim_dyn(
                i, optimizers.optimizers[group], deleted_mask, param
            )  #
        torch.cuda.empty_cache()

    def remove_from_optim_dyn(self, i, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        param = optimizer.param_groups[0]["params"][i][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][i]
        optimizer.param_groups[0]["params"].insert(i, new_params)
        optimizer.state[new_params] = param_state

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            if self.config.continue_split_dyn:
                with torch.no_grad():
                    visible_mask_dyn = (
                        self.radii_dyn[str(self.camera_idx)] > 0
                    ).flatten()
                    grads_dyn = (
                        self.xys_dyn[str(self.camera_idx)]
                        .absgrad[0][visible_mask_dyn]
                        .norm(dim=-1)
                    )  # type: ignore
                    if self.xys_grad_norm_dyn[str(self.camera_idx)] is None:
                        self.xys_grad_norm_dyn[str(self.camera_idx)] = torch.zeros(
                            self.num_points_dyn(self.camera_idx),
                            device=self.device,
                            dtype=torch.float32,
                        )  #  + self.num_points_dyn
                        self.vis_counts_dyn[str(self.camera_idx)] = torch.ones(
                            self.num_points_dyn(self.camera_idx),
                            device=self.device,
                            dtype=torch.float32,
                        )  #  + self.num_points_dyn

                    assert self.vis_counts_dyn[str(self.camera_idx)] is not None
                    self.vis_counts_dyn[str(self.camera_idx)][visible_mask_dyn] += 1
                    self.xys_grad_norm_dyn[str(self.camera_idx)][visible_mask_dyn] += (
                        grads_dyn
                    )
                    # update the max screen size, as a ratio of number of pixels
                    if self.max_2Dsize_dyn[str(self.camera_idx)] is None:
                        self.max_2Dsize_dyn[str(self.camera_idx)] = torch.zeros_like(
                            self.radii_dyn[str(self.camera_idx)], dtype=torch.float32
                        )
                    newradii_dyn = self.radii_dyn[str(self.camera_idx)].detach()[
                        visible_mask_dyn
                    ]
                    self.max_2Dsize_dyn[str(self.camera_idx)][visible_mask_dyn] = (
                        torch.maximum(
                            self.max_2Dsize_dyn[str(self.camera_idx)][visible_mask_dyn],
                            newradii_dyn
                            / float(max(self.last_size[0], self.last_size[1])),
                        )
                    )
            return

        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(
                    self.num_points, device=self.device, dtype=torch.float32
                )
                self.vis_counts = torch.ones(
                    self.num_points, device=self.device, dtype=torch.float32
                )
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

            if self.config.use_adc:
                # for dynamic points
                visible_mask_dyn = (self.radii_dyn[str(self.camera_idx)] > 0).flatten()
                grads_dyn = (
                    self.xys_dyn[str(self.camera_idx)]
                    .absgrad[0][visible_mask_dyn]
                    .norm(dim=-1)
                )  # type: ignore
                if self.xys_grad_norm_dyn[str(self.camera_idx)] is None:
                    self.xys_grad_norm_dyn[str(self.camera_idx)] = torch.zeros(
                        self.num_points_dyn(self.camera_idx),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    self.vis_counts_dyn[str(self.camera_idx)] = torch.ones(
                        self.num_points_dyn(self.camera_idx),
                        device=self.device,
                        dtype=torch.float32,
                    ) 

                assert self.vis_counts_dyn[str(self.camera_idx)] is not None
                self.vis_counts_dyn[str(self.camera_idx)][visible_mask_dyn] += 1
                self.xys_grad_norm_dyn[str(self.camera_idx)][visible_mask_dyn] += (
                    grads_dyn
                )
                # update the max screen size, as a ratio of number of pixels
                if self.max_2Dsize_dyn[str(self.camera_idx)] is None:
                    self.max_2Dsize_dyn[str(self.camera_idx)] = torch.zeros_like(
                        self.radii_dyn[str(self.camera_idx)], dtype=torch.float32
                    )
                newradii_dyn = self.radii_dyn[str(self.camera_idx)].detach()[
                    visible_mask_dyn
                ]
                self.max_2Dsize_dyn[str(self.camera_idx)][visible_mask_dyn] = (
                    torch.maximum(
                        self.max_2Dsize_dyn[str(self.camera_idx)][visible_mask_dyn],
                        newradii_dyn / float(max(self.last_size[0], self.last_size[1])),
                    )
                )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        
        keys = [
            "means",
            "scales",
            "quats",
            "features_dc",
            "features_rest",
            "opacities",
        ]
        if "embeddings" in self.gauss_params:
            keys.append("embeddings")  # Add dynamically if it exists

        return {
            name: [self.gauss_params[name]]
            for name in keys
        }


    def get_gaussian_param_groups_dyn_dict(
        self,
    ) -> Dict[str, dict[int, List[Parameter]]]:
        return {
            name: [
                self.gauss_params_dyn_dict[str(i)][name]
                for i in range(self.num_train_data)
            ]
            for name in [
                "means_dyn",
                "scales_dyn",
                "quats_dyn",
                "rgbs_dyn",
                "opacities_dyn",
            ]
        }

    def get_param_groups(self):
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        gps_dyn = self.get_gaussian_param_groups_dyn_dict()
        gps_bg = {}
        if self.config.enable_bg_model:
            assert self.bg_model is not None
            gps_bg["field_background_encoder"] = list(self.bg_model.encoder.parameters())
            gps_bg["field_background_base"] = list(self.bg_model.sh_base_head.parameters())
            gps_bg["field_background_rest"] = list(self.bg_model.sh_rest_head.parameters())

        if self.config.enable_appearance:
            gps["appearance_mlp"] = list(self.appearance_mlp.parameters())
            gps["appearance_embeddings"] = list(self.appearance_embeddings.parameters())

        return {**gps, **gps_dyn, **gps_bg}

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            if self.config.app_per_gauss:
                embeddings_crop = self.embeddings[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            if self.config.app_per_gauss:
                embeddings_crop = self.embeddings

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        # camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        if self.config.enable_appearance:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                # if self.training or self.dataparser_config.eval_train:
                self.camera_idx = camera.metadata["cam_idx"]
                appearance_embed = self.appearance_embeddings(
                    torch.tensor(self.camera_idx, device=self.device)
                )
            else:
                # appearance_embed is zero
                appearance_embed = torch.zeros(
                    self.config.appearance_embedding_dim, device=self.device
                )
            assert self.appearance_mlp is not None
            # assert self.embeddings is not None

            features = torch.cat(
                (features_dc_crop.unsqueeze(1), features_rest_crop), dim=1
            )
            # offset, mul = self.appearance_mlp(appearance_embed.repeat(self.num_points, 1), features_dc_crop)
            offset, mul = self.appearance_mlp(
                self.embeddings if self.config.app_per_gauss else None,
                appearance_embed.repeat(self.num_points, 1),
                features_dc_crop,
            )
            colors_toned = colors_crop * mul.unsqueeze(1) + offset.unsqueeze(1)
            shdim = (self.config.sh_degree + 1) ** 2
            colors_toned = colors_toned.view(-1, shdim, 3).contiguous().clamp_max(1.0)
            # colors_toned = eval_sh(self.active_sh_degree, colors_toned, dir_pp_normalized)
            colors_toned = torch.clamp_min(colors_toned + 0.5, 0.0)
            colors_crop = colors_toned

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
            bg_sh_degree_to_use = min(
                self.step // (self.config.sh_degree_interval // 2),
                self.config.bg_sh_degree,
            )
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render_3d, alpha_3d, info_3d = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB+ED",  #  render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info_3d["means2d"].requires_grad:
            info_3d["means2d"].retain_grad()

        self.xys = info_3d["means2d"]  # [1, N, 2]
        self.radii = info_3d["radii"][0]  # [N]

        alpha_3d = alpha_3d[:, ...]

        # Only need for one time
        ### BACKGROUND MODEL
        if self.config.enable_bg_model:
            directions = normalize(
                camera.generate_rays(camera_indices=0, keep_shape=False).directions
            )

            bg_sh_coeffs = self.bg_model.get_sh_coeffs(
                appearance_embedding=appearance_embed,
            )

            background = spherical_harmonics(
                degrees_to_use=bg_sh_degree_to_use,
                coeffs=bg_sh_coeffs.repeat(directions.shape[0], 1, 1),
                dirs=directions,
            )
            background = background.view(H, W, 3)
        else:
            background = self._get_background_color().view(1, 1, 3)

        rgb_static = render_3d[:, ..., :3] + (1 - alpha_3d) * background

        rgb_static = torch.clamp(rgb_static, 0.0, 1.0)

        rgb = rgb_static

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        returns = {}
        returns["rgb"] = rgb.squeeze(0)
        returns["depth"] = render_3d[..., 3].squeeze(0).unsqueeze(-1)
        returns["background"] = background
        returns["accumulation"] = alpha_3d.squeeze(0)

        img_dyn, alpha_2d, depth_2d = None, None, None  # for debug
        if camera.metadata is not None and "cam_idx" in camera.metadata:
            self.camera_idx = camera.metadata["cam_idx"]

            dyn_gaussians = self.gauss_params_dyn_dict[str(self.camera_idx)]
            opacities_dyn = dyn_gaussians["opacities_dyn"]
            means_dyn = dyn_gaussians["means_dyn"]
            rgbs_dyn = dyn_gaussians["rgbs_dyn"]
            scales_dyn = dyn_gaussians["scales_dyn"]
            quats_dyn = dyn_gaussians["quats_dyn"]

            colors_dyn = rgbs_dyn.clamp_min(0.0).clamp_max(1.0)

            render_2d, alpha_2d, info_2d = rasterization(
                means=means_dyn,
                quats=quats_dyn / quats_dyn.norm(dim=-1, keepdim=True),
                scales=torch.exp(scales_dyn),
                opacities=torch.sigmoid(opacities_dyn).squeeze(-1),
                colors=colors_dyn,  # [N, 3]
                viewmats=viewmat,  # [C, 4, 4]
                Ks=K,  # [C, 3, 3]
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                render_mode="RGB+ED",
                sh_degree=None,  # sh_degree_to_use,
                packed=False,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.config.rasterize_mode,
            )
            if info_2d["means2d"].requires_grad:
                info_2d["means2d"].retain_grad()
            if self.config.use_adc:
                self.xys_dyn[str(self.camera_idx)] = info_2d["means2d"]  # [1, N, 2]
                self.radii_dyn[str(self.camera_idx)] = info_2d["radii"][0]  # [N]
            depth_2d = render_2d[..., 3].unsqueeze(-1)  # [H, W, 1]
            render_2d = render_2d[..., :3]  # [H, W, 3]
            img_dyn = render_2d.squeeze(0)
            img_dyn = torch.clamp(img_dyn, 0.0, 1.0)

            rgb = (
                render_2d
                + (1 - alpha_2d) * render_3d[:, ..., :3]
                + (1 - alpha_3d) * (1 - alpha_2d) * background
            )
            rgb = torch.clamp(rgb, 0.0, 1.0)
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
            alpha_2d = alpha_2d.squeeze(0)
            depth_2d = depth_2d.squeeze(0)

            returns["img_dyn"] = img_dyn
            returns["alpha_2d"] = alpha_2d
            returns["depth_2d"] = depth_2d
            returns["rgb"] = rgb.squeeze(0)
            returns["rgb_static"] = rgb_static.squeeze(0)
        else:
            self.camera_idx = None

        return returns

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count_static"] = self.num_points
        # metrics_dict["gaussian_count_transient"] = self.num_points_dyn

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        Ll1_img = torch.abs(gt_img - pred_img)

        Ll1 = Ll1_img.mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )
        main_loss = (
            1 - self.config.ssim_lambda
        ) * Ll1 + self.config.ssim_lambda * simloss

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": main_loss,
            "scale_reg": scale_reg,
        }

        # add loss to prevent the hole of static
        if self.config.alpha_bg_loss_lambda > 0:
            if self.config.enable_bg_model:
                alpha_loss = torch.tensor(0.0).to(self.device)
                background = outputs["background"]
                alpha = outputs["alpha_2d"].detach() + (1 - outputs["alpha_2d"].detach()) * outputs["accumulation"]
                # alpha = outputs["accumulation"]
                # for those pixel are well represented by bg and has low alpha, we encourage the gaussian to be transparent
                bg_mask = torch.abs(gt_img - background).mean(dim=-1, keepdim=True) < 0.003
                # use a box filter to avoid penalty high frequency parts
                f = 3
                window = (torch.ones((f, f)).view(1, 1, f, f) / (f * f)).cuda()
                bg_mask = (
                    torch.nn.functional.conv2d(
                        bg_mask.float().unsqueeze(0).permute(0, 3, 1, 2),
                        window,
                        stride=1,
                        padding="same",
                    )
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                )
                alpha_mask = bg_mask > 0.6
                # prevent NaN
                if alpha_mask.sum() != 0:
                    alpha_loss = alpha[alpha_mask].mean() * self.config.alpha_bg_loss_lambda # default: 0.15
                else:
                    alpha_loss = torch.tensor(0.0).to(self.device)
                loss_dict["alpha_bg_loss"] = alpha_loss
            else:
                alpha_bg = outputs["accumulation"]
                loss_dict["alpha_bg_loss"] = self.config.alpha_bg_loss_lambda * (1 - alpha_bg).mean()

        if self.config.alpha_2d_loss_lambda > 0 and "alpha_2d" in outputs:
            alpha_2d = outputs["alpha_2d"]
            loss_dict["alpha_2d_loss"] = self.config.alpha_2d_loss_lambda * alpha_2d.mean()

        return loss_dict

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        predicted_rgb = outputs["rgb"]

        assert gt_rgb.shape == predicted_rgb.shape
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
        }

        return metrics_dict, images_dict

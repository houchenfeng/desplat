import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Dict, List, Literal, Optional, Type

import torch
import torch._dynamo
import torch.distributed as dist
import torchvision.utils as vutils
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from desplat.datamanager import (
    DeSplatDataManager,
    DeSplatDataManagerConfig,
)
from desplat.desplat_model import DeSplatModel, DeSplatModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler

torch._dynamo.config.suppress_errors = True


@dataclass
class DeSplatPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: DeSplatPipeline)
    """target class to instantiate"""

    datamanager: DataManagerConfig = field(
        default_factory=lambda: DeSplatDataManagerConfig()
    )
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=lambda: DeSplatModelConfig())
    """specifies the model config"""
    # finetune: bool = True
    # """Whether to mask the left half and evaluate the right half of the images"""
    test_time_optimize: bool = False


class DeSplatPipeline(VanillaPipeline):
    def __init__(
        self,
        config: DeSplatPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DeSplatDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)

        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata  # type: ignore
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]  # type: ignore
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]  # type: ignore

            seed_pts = (pts, pts_rgb)

        # if not config.model.finetune:
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
            cameras=self.datamanager.train_dataset.cameras,
            dataparser=self.datamanager.dataparser_config,
        )

        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                DeSplatModel,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}  # , **model_params_dyn

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Evaluating all images...", total=num_images
            )
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(
                    outputs, batch
                )
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(),
                            output_path / f"{image_prefix}_{key}_{idx:04d}.png",
                        )

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (
                    num_rays / (time() - inner_start)
                ).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (
                    metrics_dict["num_rays_per_sec"] / (height * width)
                ).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor(
                        [metrics_dict[key] for metrics_dict in metrics_dict_list]
                    )
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor(
                            [metrics_dict[key] for metrics_dict in metrics_dict_list]
                        )
                    )
                )

        if self.test_mode == "inference":
            print("Now we are in the test mode.")
            metrics_dict["full_results"] = metrics_dict_list
            del image_dict["depth"]

        self.train()
        return metrics_dict

    @profiler.time_function
    def get_average_eval_half_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Get the average metrics for evaluation images."""
        # assert hasattr(
        #     self.datamanager, "fixed_indices_eval_dataloader"
        # ), "datamanager must have 'fixed_indices_eval_dataloader' attribute"
        image_prefix = "eval"
        return self.get_average_half_image_metrics(
            self.datamanager.fixed_indices_eval_dataloader,
            image_prefix,
            step,
            output_path,
            get_std,
        )

    @profiler.time_function
    def get_average_half_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Evaluating all images...", total=num_images
            )
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width

                half_width = width // 2
                for key, img in outputs.items():
                    if img.dim() == 3:  # (H, W, C)
                        masked_img = img.clone()
                        masked_img = img[:, half_width:, :].clone()
                        outputs[key] = masked_img
                right_half = {}
                right_half["image"] = batch["image"][:, half_width:, :]

                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(
                    outputs, right_half
                )
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(),
                            output_path / f"{image_prefix}_{key}_{idx:04d}.png",
                        )

                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor(
                        [metrics_dict[key] for metrics_dict in metrics_dict_list]
                    )
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor(
                            [metrics_dict[key] for metrics_dict in metrics_dict_list]
                        )
                    )
                )

        self.train()
        return metrics_dict

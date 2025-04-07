import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Union

import torch
import torch._dynamo

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)

torch._dynamo.config.suppress_errors = True


@dataclass
class DeSplatDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: DeSplatDataManager)


class DeSplatDataManager(FullImageDatamanager):
    config: DeSplatDataManagerConfig

    def __init__(
        self,
        config: DeSplatDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        metadata = self.train_dataparser_outputs.metadata
        if test_mode == "test":
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = [d.copy() for d in self.cached_eval]
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            if (
                self.dataparser_config.eval_train
                or self.dataparser_config.test_time_optimize
            ):
                if _cameras.metadata is None:
                    _cameras.metadata = {}
                _cameras.metadata["cam_idx"] = i
            cameras.append(_cameras[i : i + 1])

        assert (
            len(self.eval_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        return list(zip(cameras, data))

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle

        TODO: Make sure this logic is consistent with the vanilladatamanager"""
        image_idx = self.eval_unseen_cameras.pop(
            random.randint(0, len(self.eval_unseen_cameras) - 1)
        )
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = self.cached_eval[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)
        assert (
            len(self.eval_dataset.cameras.shape) == 1
        ), "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        # keep metadata for debugging
        if self.dataparser_config.eval_train:
            if camera.metadata is None:
                camera.metadata = {}
            camera.metadata["cam_idx"] = image_idx
        return camera, data

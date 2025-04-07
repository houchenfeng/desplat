# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phototourism dataset parser. Datasets and documentation here: http://phototour.cs.washington.edu/datasets/"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Type

import numpy as np
import torch

from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
)

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.plugins.registry_dataparser import DataParserSpecification


@dataclass
class RobustNerfDataParserConfig(ColmapDataParserConfig):
    """On-the-go dataset parser config"""

    _target: Type = field(default_factory=lambda: RobustNerfDataParser)
    """target class to instantiate"""
    eval_train: bool = False
    """evaluate test set or train set, for debug"""
    train_split_mode: Optional[Literal["ratio", "number", "filename"]] = None
    """How to split the training images. If None, all cluttered images are used."""
    train_split_clean_clutter_ratio: float = 1.0
    """The percentage of the training images that are cluttered. 0.0 -> only clean images, 1.0 -> only cluttered images"""
    train_split_clean_clutter_number: int = 0
    """The number of clean images to use for training. If 0, all clean images are used."""
    idx_clutter: List[int] = field(default_factory=lambda: [76])
    """The indices of the cluttered images to use for training"""
    colmap_path: Path = Path("sparse/0")
    """path to colmap sparse folder"""
    downscale_factor: int = 8
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    test_time_optimize: bool = False
    """Whether to use test-time optimization for the dataset"""


@dataclass
class RobustNerfDataParser(ColmapDataParser):
    """RobustNerf dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: RobustNerfDataParserConfig

    def __init__(self, config: ColmapDataParserConfig):
        super().__init__(config=config)
        self.config = config
        self.data = config.data

    def _get_image_indices(self, image_filenames, split):
        i_train, i_eval = self.get_train_eval_split_filename(
            image_filenames, self.config.train_split_clean_clutter_ratio
        )

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        return indices

    def get_train_eval_split_filename(
        self, image_filenames: List, clean_clutter_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the train/eval split based on the filename of the images.

        Args:
            image_filenames: list of image filenames
        """

        num_images = len(image_filenames)
        basenames = [
            os.path.basename(image_filename) for image_filename in image_filenames
        ]

        i_all = np.arange(num_images)
        i_clean = []
        i_clutter = []

        i_train_clean = []
        i_train_clutter = []

        i_train = []
        i_eval = []

        for idx, basename in zip(i_all, basenames):
            if "clean" in basename:
                i_clean.append(idx)

            elif "clutter" in basename:
                i_clutter.append(idx)
                
            elif "extra" in basename:
                i_eval.append(idx)  # extra is always used as eval
            else:
                raise ValueError(
                    "image frame should contain clutter/extra in its name "
                )

        if self.config.train_split_mode is None:
            i_train = i_clutter
            if self.config.eval_train:
                i_eval = i_train
            return np.array(i_train), np.array(i_eval)

        if len(i_clean) > len(i_clutter):
            i_clean = i_clean[: len(i_clutter)]
        elif len(i_clean) < len(i_clutter):
            i_clutter = i_clutter[: len(i_clean)]
        num_images_train = min(len(i_clean), len(i_clutter))

        print("Number of clean images: ", num_images_train)

        if self.config.train_split_mode == "number":
            i_perm = torch.randperm(
                num_images_train, generator=torch.Generator().manual_seed(2023)
            ).tolist()
            num_images_cluttered = self.config.train_split_clean_clutter_number
            i_train = []
            i_train_clutter = []
            # loop over permuted indices to select one image from each clean/clutter pair

            for k, idx in enumerate(i_perm):
                if k < num_images_cluttered:
                    i_train_clutter.append(i_clutter[idx])
                else:
                    i_train.append(i_clean[idx])
            i_train.expand(i_train_clutter)
        elif self.config.train_split_mode == "ratio":
            i_train_clutter = []
            if clean_clutter_ratio == 0.0:
                # only clean images
                i_train = i_clean
            elif clean_clutter_ratio == 1.0:
                # only cluttered images
                i_train = i_clutter
            elif clean_clutter_ratio > 0.0 and clean_clutter_ratio < 1.0:
                # pick either clutter/clean image once
                i_perm = torch.randperm(
                    num_images_train, generator=torch.Generator().manual_seed(2023)
                ).tolist()
                num_images_cluttered = int(
                    num_images_train * clean_clutter_ratio
                )  # rounds down
                i_train = []
                # loop over permuted indices to select one image from each clean/clutter pair
                for k, idx in enumerate(i_perm):
                    if k < num_images_cluttered:
                        i_train_clutter.append(i_clutter[idx])
                    else:
                        i_train.append(i_clean[idx])
                i_train.extend(i_train_clutter)
                print(
                    "basenames of cluttered images: ",
                    [basenames[i] for i in i_train_clutter],
                )
            else:
                raise ValueError(
                    "arg train_split_clean_clutter_ratio must be between 0.0 and 1.0 "
                )

        elif self.config.train_split_mode == "filename":
            # manually select images
            i_train = i_clean
            i_train.extend(i_train_clutter)

            # Remove elements in i_train_clean from i_train
            for item in i_train_clean:
                if item in i_train:
                    i_train.remove(item)
            print(
                "basenames of cluttered images: ",
                [basenames[i] for i in i_train_clutter],
            )
        else:
            raise ValueError("Unknown train_split_mode")
        print("i_train", i_train)
        print("i_eval", i_eval)
        if self.config.eval_train:
            i_eval = i_train
        return np.array(i_train), np.array(i_eval)


RobustNerfDataParserSpecification = DataParserSpecification(
    config=RobustNerfDataParserConfig(),
    description="RobustNeRF dataparser",
)

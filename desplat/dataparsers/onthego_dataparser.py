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

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np

from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
)

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.plugins.registry_dataparser import DataParserSpecification


@dataclass
class OnthegoDataParserConfig(ColmapDataParserConfig):
    """On-the-go dataset parser config"""

    _target: Type = field(default_factory=lambda: OnthegoDataParser)
    """target class to instantiate"""
    eval_train: bool = False
    """evaluate test set or train set, for debug"""
    colmap_path: Path = Path("sparse/0")
    """path to colmap sparse folder"""
    test_time_optimize: bool = False
    """Whether to use test-time optimization for the dataset"""


@dataclass
class OnthegoDataParser(ColmapDataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: OnthegoDataParserConfig

    def __init__(self, config: ColmapDataParserConfig):
        super().__init__(config=config)
        self.config = config
        self.data: Path = config.data
        self.config.downscale_factor = (
            4 if self.data.name == "patio" or self.data.name == "arcdetriomphe" else 8
        )
        if self.data.name == "data_0.3":
            self.config.downscale_factor = (
           2
        )
        print("self.data.name ",self.data.name ," self.config.downscale_factor ",self.config.downscale_factor)
    def _get_image_indices(self, image_filenames, split):
        # Load the split file to get the train/eval split
        if os.path.exists(os.path.join(self.config.data, "split.json")):
            with open(os.path.join(self.config.data, "split.json"), "r") as file:
                split_json = json.load(file)

            # Select the split according to the split file.
            all_indices = np.arange(len(image_filenames))

            i_eval = all_indices[split_json["extra"]]

            i_train = all_indices[split_json["clutter"]]

            if self.config.eval_train:
                i_eval = i_train
            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")
        else:
            # If no split file is provided, use all images for training
            indices = np.arange(len(image_filenames))
        return indices


OnthegoDataParserSpecification = DataParserSpecification(
    config=OnthegoDataParserConfig(),
    description="On-the-go dataparser",
)

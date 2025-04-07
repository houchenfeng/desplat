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

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np

# from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import (
    ColmapDataParser,
    ColmapDataParserConfig,
)
from nerfstudio.plugins.registry_dataparser import DataParserSpecification


@dataclass
class PhotoTourismDataParserConfig(ColmapDataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: PhotoTourismDataParser)
    """target class to instantiate"""
    eval_train: bool = False
    """evaluate test set or train set, for debug"""
    colmap_path: Path = Path("sparse")

    test_time_optimize: bool = True


@dataclass
class PhotoTourismDataParser(ColmapDataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: PhotoTourismDataParserConfig

    def __init__(self, config: PhotoTourismDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data

    def _get_image_indices(self, image_filenames, split):
        # Load the split file to get the train/eval split
        if "brandenburg_gate" in str(self.data):
            tsv_path = self.config.data / "brandenburg.tsv"
        elif "sacre_coeur" in str(self.data):
            tsv_path = self.config.data / "sacre.tsv"
        elif "trevi_fountain" in str(self.data):
            tsv_path = self.config.data / "trevi.tsv"
        else:
            raise ValueError(f"Unknown dataset {self.data.name}")

        basenames = [
            os.path.basename(image_filename) for image_filename in image_filenames
        ]

        train_names, test_names = set(), set()

        with open(tsv_path, newline="") as tsv_file:
            reader = csv.reader(tsv_file, delimiter="\t")
            next(reader)
            for row in reader:
                if row[2] == "train":
                    train_names.add(row[0])
                elif row[2] == "test":
                    test_names.add(row[0])

        if self.config.eval_train:
            split = "train"

        indices = [
            idx
            for idx, basename in enumerate(basenames)
            if (basename in train_names and split == "train")
            or (basename in test_names and split in ["val", "test"])
        ]

        if not indices and split not in ["train", "test", "val"]:
            raise ValueError(f"Unknown dataparser split {split}")

        return np.array(indices)


PhotoTourismDataParserSpecification = DataParserSpecification(
    config=PhotoTourismDataParserConfig(), description="Photo tourism dataparser"
)

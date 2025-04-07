from __future__ import annotations

from pathlib import Path

from desplat.datamanager import DeSplatDataManagerConfig
from desplat.dataparsers.robustnerf_dataparser import RobustNerfDataParserConfig
from desplat.desplat_model import DeSplatModelConfig
from desplat.pipeline import DeSplatPipelineConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

desplat_method = MethodSpecification(
    config=TrainerConfig(
        method_name="desplat",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=30000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=DeSplatPipelineConfig(
            datamanager=DeSplatDataManagerConfig(  # desplat
                dataparser=RobustNerfDataParserConfig(
                    load_3D_points=True, colmap_path=Path("sparse/0")
                ),  # , downscale_factor=2
                cache_images_type="uint8",
            ),
            model=DeSplatModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "embeddings": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-15),
                "scheduler": None,
            },
            "means_dyn": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "rgbs_dyn": {
                "optimizer": AdamOptimizerConfig(lr=0.025, eps=1e-15),
                "scheduler": None,
            },
            "opacities_dyn": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales_dyn": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "quats_dyn": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            # back to original components
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "appearance_mlp": {
                "optimizer": AdamOptimizerConfig(lr=0.0005, eps=1e-15),
                "scheduler": None,
            },
            "appearance_embeddings": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
            "field_background_encoder": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=100000
                ),
            },
            "field_background_base": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4, max_steps=100000
                ),
            },
            "field_background_rest": {
                "optimizer": AdamOptimizerConfig(lr=2e-3 / 20, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=2e-4 / 20, max_steps=100000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Desplat",
)

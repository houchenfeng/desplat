import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch
import yaml

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE


def load_checkpoint_for_eval(
    config: TrainerConfig, pipeline: Pipeline
) -> Tuple[Path, int]:
    """Load a checkpointed pipeline for evaluation.

    Args:
        config (TrainerConfig): Configuration for loading the pipeline.
        pipeline (Pipeline): The pipeline instance to load weights into.

    Returns:
        Tuple containing the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None, "Checkpoint directory must be specified."
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from specified directory.")
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(
                f"Checkpoint directory not found at {config.load_dir}", justify="center"
            )
            CONSOLE.print(
                "Ensure checkpoints were generated during training.", justify="center"
            )
            sys.exit(1)
        load_step = max(
            int(f.split("-")[1].split(".")[0])
            for f in os.listdir(config.load_dir)
            if "step-" in f
        )
    else:
        load_step = config.load_step

    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist."

    # Load checkpoint
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Successfully loaded checkpoint from {load_path}")

    return load_path, load_step


def setup_evaluation(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int, float]:
    """Set up pipeline loading for evaluation, with an option to calculate model size.

    Args:
        config_path: Path to the configuration YAML file.
        eval_num_rays_per_chunk: Rays per forward pass (optional).
        test_mode: Data loading mode ('test', 'val', or 'inference').
        update_config_callback: Optional function to modify config before loading the pipeline.

    Returns:
        Config, loaded pipeline, checkpoint path, step, and model size in MB.
    """
    # Load and validate configuration
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[
        config.method_name
    ].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback:
        config = update_config_callback(config)

    # Define the checkpoint directory
    config.load_dir = config.get_checkpoint_dir()

    # Initialize the pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    pipeline.eval()

    # Load the checkpoint
    checkpoint_path, step = load_checkpoint_for_eval(config, pipeline)

    # Calculate the size of the loaded model
    model_size_mb = calculate_model_size(pipeline)
    CONSOLE.print(f"Model size: {model_size_mb:.2f} MB")

    return config, pipeline, checkpoint_path, step, model_size_mb


def calculate_model_size(model: torch.nn.Module) -> float:
    """Calculate the size of a PyTorch model in MB.

    Args:
        model: The PyTorch model for which to calculate the memory usage.

    Returns:
        Model size in megabytes (MB).
    """
    dynamic_param_size = 0
    static_param_size = 0

    for name, p in model.named_parameters():
        # Determine if the parameter is dynamic or static
        if "gauss_params_dyn_dict" in name:
            dynamic_param_size += p.nelement() * p.element_size()
        else:
            static_param_size += p.nelement() * p.element_size()

    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())

    # Print size of each parameter category
    print(
        "Total dynamic parameters size: {:.3f} MB".format(dynamic_param_size / 1024**2)
    )
    print("Total static parameters size: {:.3f} MB".format(static_param_size / 1024**2))
    print(
        "Total model size: {:.3f} MB".format(
            (dynamic_param_size + static_param_size) / 1024**2
        )
    )

    total_size_mb = (
        dynamic_param_size + static_param_size + buffer_size
    ) / 1024**2  # Convert to MB
    return total_size_mb


if __name__ == "__main__":
    import tyro

    tyro.cli(setup_evaluation)

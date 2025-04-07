"""test time camera appearance optimization

Usage:

python scripts/test_time_optimize.py --load-config [path_to_script]
"""

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image
from torchvision.utils import save_image

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup


@dataclass
class AppearanceModelConfigs:
    # TODO: only works when there are no per-gauss features atm
    app_per_gauss: bool = False
    appearance_embedding_dim: int = 32
    appearance_n_fourier_freqs: int = 4
    appearance_init_fourier: bool = True


@dataclass
class TestTimeOpt:
    load_config: Path = Path("")
    """Path to the config YAML file."""
    train_iters: int = 128

    """train iters"""
    save_gif: bool = False
    """save a training gif"""
    lr_app_emb: float = 0.01
    """learning rate for appearance embedding"""
    metrics_output_path: Path = Path("./test_time_metrics/")
    """Output path of test time opt eval metrics"""
    use_saved_embedding: bool = False
    """Use saved embedding module"""
    save_all_imgs: bool = False
    """Save all images"""

    def main(self):
        if "brandenburg_gate" in str(self.load_config) or "unnamed" in str(self.load_config):
            scene = "brandenburg_gate"
        elif "sacre_coeur" in str(self.load_config):
            scene = "sacre_coeur"
        elif "trevi_fountain" in str(self.load_config):
            scene = "trevi_fountain"
        else:
            raise ValueError(f"Unknown dataset")

        if not self.metrics_output_path.exists():
            self.metrics_output_path.mkdir(parents=True)

        config, pipeline, _, _ = eval_setup(self.load_config)
        pipeline.test_time_optimize = True
        pipeline.train()
        pipeline.cuda()
        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        train_dataset: InputDataset = pipeline.datamanager.train_dataset
        eval_dataset: InputDataset = pipeline.datamanager.eval_dataset
        cameras: Cameras = pipeline.datamanager.eval_dataset.cameras  # type: ignore
        # init app model
        app_config = AppearanceModelConfigs()
        
        if self.use_saved_embedding:
            model.appearance_embeddings = torch.load("embedding_"+ scene + ".pth")
        
        else:
            # define app model optimizers
            model.appearance_embeddings = torch.nn.Embedding(
                len(eval_dataset), app_config.appearance_embedding_dim
            ).cuda()
            model.appearance_embeddings.weight.data.normal_(0, 0.01)

            optimizer = torch.optim.Adam(
                model.appearance_embeddings.parameters(),
                lr=self.lr_app_emb,
            )

            # Force model to have appearance
            model.config.enable_appearance = True
        
            # train eval dataset
            gif_frames = []
            
            # before test time metrics:
            before_test_time_metrics = pipeline.get_average_eval_half_image_metrics(
                step=0, output_path=None
            )
            print("Metrics before test-time: ", before_test_time_metrics)

            for epoch in range(self.train_iters):
                for image_idx, data in enumerate(
                    pipeline.datamanager.cached_eval  # Undistorted images
                ):  # type: ignore
                    # process batch gt data
                    # process pred outputs
                    camera = cameras[image_idx : image_idx + 1]
                    camera.metadata = {}
                    camera.metadata["cam_idx"] = image_idx
                    camera = camera.to("cuda")

                    height, width = camera.height, camera.width

                    outputs = model.get_outputs(camera=camera)  # type: dict
                    outputs_left_half = {}
                    # mask the right half of the image
                    for key, img in outputs.items():
                        half_width = width // 2
                        if key == 'background':
                            outputs_left_half[key] = img
                        if img.dim() == 3:  # (H, W, C)
                            masked_img = img[:, :half_width, :].clone()
                            
                            # masked_img[:, half_width:, :] = 0
                            outputs_left_half[key] = masked_img
                    left_half = {}
                    left_half["image"] = data["image"][:, :half_width, :]

                    loss_dict = model.get_loss_dict(outputs=outputs_left_half, batch=left_half)
                    if image_idx == 1:
                        # choose the right side of the image
                        rgb = outputs["rgb"][:, width // 2:, :]
                        gt_img = data["image"][:, width // 2:, :]
                        save_image(rgb.permute(2, 0, 1), "rgb.jpg")
                        print("Epoch: ", epoch, "loss of img_0:", loss_dict["main_loss"])
                        if self.save_gif and epoch % 1 == 0:
                            gif_frames.append(
                                (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                            )

                    loss = functools.reduce(torch.add, loss_dict.values())
                    loss.backward()
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # save pth
            torch.save(model.appearance_embeddings, "embedding_"+ scene + ".pth")

        # Get eval metrics after
        after_test_time_metrics = pipeline.get_average_eval_half_image_metrics(
            step=0, output_path=None
        )

        print("Metrics after test-time: ", after_test_time_metrics)
        
        output_dir = f"{self.metrics_output_path}/{scene}"
        
        os.makedirs(output_dir, exist_ok=True)

        metrics_path = f"{output_dir}/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(after_test_time_metrics, f)
        
        if self.save_gif:
            gif_frames = [Image.fromarray(frame) for frame in gif_frames]
            out_dir = os.path.join(os.getcwd(), f"renders/{scene}")
            os.makedirs(out_dir, exist_ok=True)
            print(f"saving depth gif to {out_dir}/training.gif")
            gif_frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=gif_frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

        if self.save_all_imgs:
            for image_idx, data in enumerate(
                    pipeline.datamanager.cached_eval  # Undistorted images
                ):  # type: ignore
                    # process batch gt data
                    # process pred outputs
                    camera = cameras[image_idx : image_idx + 1]
                    camera.metadata = {}
                    camera.metadata["cam_idx"] = image_idx
                    camera = camera.to("cuda")

                    height, width = camera.height, camera.width

                    outputs = model.get_outputs(camera=camera)  # type: dict

                    # Define the output directory
                    out_dir = os.path.join(os.getcwd(), f"renders/{scene}")

                    # Create the directory if it doesn't exist
                    os.makedirs(out_dir, exist_ok=True)

                    # Define the full path for the image file
                    image_path = os.path.join(out_dir, f"render_{image_idx}.jpg")

                    # out_dir = os.path.join(os.getcwd(), f"renders/{scene}/render_{image_idx}.jpg")
                    save_image(outputs["rgb"].permute(2, 0, 1), image_path)

if __name__ == "__main__":
    tyro.cli(TestTimeOpt).main()

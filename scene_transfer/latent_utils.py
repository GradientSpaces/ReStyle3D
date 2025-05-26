from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image

from scene_transfer_model import SceneTransfer
from scene_transfer.config import RunConfig
from scene_transfer import image_utils
from scene_transfer.ddpm_inversion import invert
from scene_transfer.depth_estimator import get_depthmaps
from utils.logging import logger

def load_latents_or_invert_images(model: Union[SceneTransfer], cfg: RunConfig):
    if cfg.load_latents and cfg.app_latent_save_path.exists() and cfg.struct_latent_save_path.exists():
        logger.info("Loading existing latents...")
        latents_app, latents_struct = load_latents(cfg.app_latent_save_path, cfg.struct_latent_save_path)
        noise_app, noise_struct = load_noise(cfg.app_latent_save_path, cfg.struct_latent_save_path)
    else:
        logger.info("Inverting images...")
        app_image, struct_image = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        # Load depth images
        if cfg.pred_depth:
            depth_app, depth_struct = get_depthmaps(cfg)
            depth_app, depth_struct = Image.fromarray(depth_app).convert("RGB"), Image.fromarray(depth_struct).convert("RGB")
            depth_app.save(cfg.app_depth_path)
            depth_struct.save(cfg.struct_depth_path)
        else:
            depth_app = Image.open(cfg.app_depth_path).convert("RGB")
            depth_struct = Image.open(cfg.struct_depth_path).convert("RGB")
        
        # Ensure depth images are the same size as the input images
        depth_app = depth_app.resize(app_image.shape[:2])
        depth_struct = depth_struct.resize(struct_image.shape[:2])

        # Normalize depth images to [0, 1]
        depth_app = np.array(depth_app).astype(np.float32) / 255.0
        depth_struct = np.array(depth_struct).astype(np.float32) / 255.0

        # Convert back to PIL Image
        depth_app = Image.fromarray((depth_app * 255).astype(np.uint8))
        depth_struct = Image.fromarray((depth_struct * 255).astype(np.uint8))
        
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents_app, latents_struct, noise_app, noise_struct = invert_images(app_image=app_image,
                                                                             struct_image=struct_image,
                                                                             sd_model=model.pipe,
                                                                             depth_app=depth_app,
                                                                             depth_struct=depth_struct,
                                                                             cfg=cfg)
        model.enable_edit = True
    return latents_app, latents_struct, noise_app, noise_struct


def load_latents(app_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_app = torch.load(app_latent_save_path, weights_only=True)
    latents_struct = torch.load(struct_latent_save_path, weights_only=True)
    if type(latents_struct) == list:
        latents_app = [l.to("cuda") for l in latents_app]
        latents_struct = [l.to("cuda") for l in latents_struct]
    else:
        latents_app = latents_app.to("cuda")
        latents_struct = latents_struct.to("cuda")
    return latents_app, latents_struct


def load_noise(app_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_app = torch.load(app_latent_save_path.parent / (app_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_struct = torch.load(struct_latent_save_path.parent / (struct_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_app = latents_app.to("cuda")
    latents_struct = latents_struct.to("cuda")
    return latents_app, latents_struct


def invert_images(sd_model: Union[SceneTransfer], app_image: Image.Image, struct_image: Image.Image, depth_app: Image.Image, depth_struct: Image.Image, cfg: RunConfig):
    input_app = torch.from_numpy(np.array(app_image)).float() / 127.5 - 1.0
    input_struct = torch.from_numpy(np.array(struct_image)).float() / 127.5 - 1.0
    
    zs_app, latents_app = invert(x0=input_app.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5,
                                 depth=depth_app)
    
    zs_struct, latents_struct = invert(x0=input_struct.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5,
                                       depth=depth_struct)
    
    # Save the inverted latents and noises
    torch.save(latents_app, cfg.latents_path / f"{cfg.app_image_path.stem}.pt")
    torch.save(latents_struct, cfg.latents_path / f"{cfg.struct_image_path.stem}.pt")
    torch.save(zs_app, cfg.latents_path / f"{cfg.app_image_path.stem}_ddpm_noise.pt")
    torch.save(zs_struct, cfg.latents_path / f"{cfg.struct_image_path.stem}_ddpm_noise.pt")
    return latents_app, latents_struct, zs_app, zs_struct


def get_init_latents_and_noises(model: Union[SceneTransfer], cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_struct.dim() == 4 and model.latents_app.dim() == 4 and model.latents_app.shape[0] > 1:
        model.latents_struct = model.latents_struct[cfg.skip_steps]
        model.latents_app = model.latents_app[cfg.skip_steps]
    init_latents = torch.stack([model.latents_struct, model.latents_app, model.latents_struct])
    init_zs = [model.zs_struct[cfg.skip_steps:], model.zs_app[cfg.skip_steps:], model.zs_struct[cfg.skip_steps:]]
    return init_latents, init_zs

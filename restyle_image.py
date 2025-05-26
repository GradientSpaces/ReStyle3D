"""
ReStyle3D: Scene-level Appearance Transfer with Semantic Correspondences

Author: Liyuan Zhu (liyzhu@stanford.edu)
Copyright (c) 2025 Stanford University
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
from typing import List
from pathlib import Path
import time
import gc

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

# Ensure all paths are properly added
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR.parent))

from scene_transfer_model import SceneTransfer
from scene_transfer.config import RunConfig, Range
from scene_transfer import latent_utils
from scene_transfer.latent_utils import load_latents_or_invert_images
from scene_transfer.semantic_matching import match_semantic_labels
from scene_transfer.model_utils import get_refining_pipe
from scene_transfer.sdxl_refiner import StableDiffusionXLControlNetPipeline as Refining_Pipe
from utils.logging import logger


class ModelManager:
    """
    Manages model pipelines for scene transfer operations.
    
    Attributes:
        transfer_pipe: SceneTransfer model for appearance transfer
        refiner_pipe: Pipeline for refining generated images
    """
    def __init__(self, config: RunConfig):
        logger.info("Initializing scene transfer models...")
        self.transfer_pipe = SceneTransfer(config)
        self.refiner_pipe = get_refining_pipe()
        logger.info("Models initialized successfully")


def run(cfg: RunConfig, pipelines: ModelManager) -> List[Image.Image]:
    """
    Main execution function for scene transfer.
    
    Args:
        cfg: Configuration parameters
        pipelines: Model pipelines
        
    Returns:
        List of generated images
    """
    # Ensure output directory exists
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = cfg.output_path / 'config.yaml'
    pyrallis.dump(cfg, open(config_path, 'w'))
    logger.info(f"Configuration saved to {config_path}")
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    logger.info(f"Random seed set to {cfg.seed}")
    
    # Get model and load latents
    model = pipelines.transfer_pipe
    logger.info("Loading or inverting latents...")
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
    logger.info("Latents loaded successfully")
    
    # Run appearance transfer
    logger.info("Running appearance transfer...")
    start_time = time.time()
    images = run_appearance_transfer(model=model, cfg=cfg, refiner_pipe=pipelines.refiner_pipe)
    elapsed = time.time() - start_time
    logger.info(f"Appearance transfer completed in {elapsed:.2f} seconds")
    
    return images
    

def run_appearance_transfer(
    model: SceneTransfer, 
    cfg: RunConfig, 
    refiner_pipe: Refining_Pipe
) -> List[Image.Image]:
    """
    Performs appearance transfer between scenes.
    
    Args:
        model: CrossSceneTransfer model
        cfg: Configuration parameters
        refiner_pipe: Pipeline for high-resolution refinement
        
    Returns:
        List of generated images (original and transferred)
    """
    logger.info("Preparing initial latents and noise...")
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    
    # Set up diffusion process
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate cross-image attention layers
    
    # Calculate attention ranges
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    logger.info(f"Cross-attention range: steps {start_step} to {end_step}")
    
    # Load depth maps
    depth_struct = Image.open(cfg.struct_depth_path)
    depth_style = Image.open(cfg.app_depth_path)
    depths = [depth_struct, depth_style, depth_struct]
    logger.info("Depth maps loaded successfully")
    
    # Load and match semantic labels
    struct_seg_dict = torch.load(cfg.struct_seg_dict, weights_only=True)
    app_seg_dict = torch.load(cfg.app_seg_dict, weights_only=True)
    matched_labels = match_semantic_labels(app_seg_dict, struct_seg_dict)
    logger.info(f"Matched {len(matched_labels)} semantic labels between scenes")
    
    # Set up semantic attenion
    model.set_multi_swap_masks(matched_labels)
    model.prepare_attn_flow()
    
    # Run diffusion with semantic attention
    logger.info(f"Starting diffusion process with {cfg.num_timesteps} steps...")
    generator = torch.Generator('cuda').manual_seed(cfg.seed)
    images = model.pipe(
        prompt=[cfg.prompt] * 3,
        image=depths,
        latents=init_latents,
        guidance_scale=1.,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=generator,
        cross_image_attention_range=Range(start=start_step, end=end_step),
        controlnet_conditioning_scale=cfg.controlnet_guidance
    ).images
    logger.info("Diffusion semantic attention completed")
    
    # High-resolution refinement
    logger.info("Starting refinement stage...")
    lowres_img = images[0] 
    refine_generator = torch.manual_seed(0)
    
    highres_img = refiner_pipe(
        prompt=["a photo of " + cfg.domain_name],
        image=lowres_img, 
        control_image=depth_struct,
        negative_prompt=['lowres, worst quality, low quality'],
        generator=refine_generator,
        width=1024, 
        height=1024,
        num_inference_steps=100,
        target_size=(1024, 1024),
        negative_target_size=(512, 512),
        strength=0.2,
        controlnet_conditioning_scale=0.8
    ).images[0]
    
    # Save outputs
    output_dir = cfg.output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    transfer_path = output_dir / "stylized.png"
    highres_img.save(transfer_path)
    logger.info(f"Transfer result saved to {transfer_path}")
    
    # Resize and concatenate images
    images = [image.resize((1024, 1024)) for image in images[1:]]
    images = [highres_img] + images
    
    joined_images = np.concatenate(images[::-1], axis=1)
    joined_path = output_dir / "joined.png"
    Image.fromarray(joined_images).save(joined_path)
    logger.info(f"Combined visualization saved to {joined_path}")
    
    return images

def generate_single_view_stylized(
    struct_img_path: Path,
    style_img_path: Path,
    struct_seg_dict: Path,
    style_seg_dict: Path,
    output_path: Path,
    scene_type: str = "bedroom",
    domain_name: str = None,
    seed: int = 42
):
    """
    Generate a stylized version of the structure image using the style image.
    Output will be saved to output_path / 'ours.png'
    """
    logger.info(f"Generating stylized image for {scene_type}")
    
    cfg = RunConfig(
        app_image_path=style_img_path,
        struct_image_path=struct_img_path,
        output_path=output_path,
        domain_name=domain_name or scene_type,
        seed=seed,
        load_latents=True,
        use_masked_adain=False
    )
    cfg.config_exp()
    
    cfg.struct_seg_dict = str(struct_seg_dict)
    cfg.app_seg_dict = str(style_seg_dict)
    
    pipes = ModelManager(cfg)
    images = run(cfg, pipes)
    
    # Save just the high-res stylized output
    output_path.mkdir(parents=True, exist_ok=True)
    stylized_output = images[0]  # High-res result from refiner
    save_path = output_path / "ours.png"
    stylized_output.save(save_path)
    
    logger.info(f"Saved single-view stylized image to {save_path}")
    
    del pipes, images, stylized_output
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_path
    
def main():
    """
    Main function to run scene transfer demonstrations.
    """
    demo_scene_types = ["bedroom", "kitchen", "living" ]
    logger.info(f"Running demos for scene types: {', '.join(demo_scene_types)}")
    
    # Initialize model
    model_cfg = RunConfig(None, None)
    pipes = ModelManager(model_cfg)
    
    for i, scene_type in enumerate(demo_scene_types):
        demo_idx = i + 1
        demo_dir = Path("demo/single_transfer") / f"pair{demo_idx}_{scene_type}"
        output_dir = Path("output") / "demo" / str(demo_idx) / "intermediate"
        
        logger.info(f"\n{'='*80}\nProcessing {scene_type} (demo {demo_idx}/3)\n{'='*80}")
        
        # Prepare paths
        style_img_path = demo_dir / "style.png"
        style_seg_dict = demo_dir / "style.pth"
        struct_img_path = demo_dir / "structure.png"
        struct_seg_dict = demo_dir / "structure.pth"

        # Check if files exist
        for path in [style_img_path, style_seg_dict, struct_img_path, struct_seg_dict]:
            if not path.exists():
                logger.error(f"Required file not found: {path}")
                return

        # Configure run
        cfg = RunConfig(
            app_image_path=style_img_path,
            output_path=output_dir,
            struct_image_path=struct_img_path,
            domain_name=scene_type,
            load_latents=True,
            use_masked_adain=False
        )
        cfg.config_exp()
        
        cfg.struct_seg_dict = str(struct_seg_dict)
        cfg.app_seg_dict = str(style_seg_dict)
        
        logger.info(f"Transferring {style_img_path.name} style to {struct_img_path.name}...")
        run(cfg, pipes)
        logger.info(f"Transfer for {scene_type} completed successfully")


if __name__ == '__main__':
    main()
import torch
from diffusers import DDIMScheduler, ControlNetModel
from diffusers import EulerAncestralDiscreteScheduler, AutoencoderKL
from typing import Optional
from scene_transfer.sd15_transfer import SemanticAttentionSD15
from scene_transfer.sdxl_refiner import StableDiffusionXLControlNetPipeline
from utils.logging import logger

def get_scene_transfer_sd15() -> SemanticAttentionSD15:
    logger.info("Loading SD1.5...")
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SemanticAttentionSD15.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                                      controlnet=ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-depth'),
                                                                      safety_checker=None).to(device)
    # pipe.unet = FreeUUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(device)
    pipe.unet.enable_freeu(s1=0.9, s2=0.2, b1=1.5, b2=1.6)
    pipe.scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    return pipe

def get_refining_pipe(precision : torch.dtype = torch.float16) -> StableDiffusionXLControlNetPipeline:
    logger.info("Loading SDXL...")
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=precision,
    )
    controlnet.enable_xformers_memory_efficient_attention()
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=precision)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=precision,
    )
    pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    
    return pipe
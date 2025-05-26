import os
import sys
import glob
import copy
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import open3d as o3d

from viewformer import ViewTransferSDXLPipeline, UNet2DConditionModel
from utils.proj_utils import project_by_splatting, compute_scaled_intrinsics
from utils.logging import logger

sys.path.append('./third_party/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


class StyleLifter:
    """ Lifting a single stylized image to multiple input views
    """
    def __init__(self, 
                 ckpt_path:str):
        self.device = "cuda" # our model can only run on GPU
        self.ckpt_path = ckpt_path
        logger.info(f"Initializing StyleLifter with checkpoint: {ckpt_path}")
        self._setup_dust3r()
        self._setup_ViewTransformer()
    
    def _setup_ViewTransformer(self):
        logger.info("Loading ViewFormer pipeline...")
        pipeline = ViewTransferSDXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",  # Add this to ensure all components load in fp16
            use_safetensors=True  # Add this for faster loading
        )
        pipeline.unet = UNet2DConditionModel.from_pretrained(
            "gradient-spaces/ReStyle3D",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipeline.to("cuda")
        pipeline.enable_xformers_memory_efficient_attention()
        
        self.ViewTransformer = pipeline

    def _setup_dust3r(self):
        logger.info("Loading DUSt3R pipeline...")
        model_path = os.path.join(self.ckpt_path, "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        self.dust3r = load_model(model_path, self.device)

    def load_initial_images(self, image_files):
        
        images = load_images(image_files, size=512, force_1024 = True)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1

        return images, img_ori
    
    def run_dust3r(self, input_images, clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=1)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene
    
    def setup_scene(self, scene_path, downsample=4):
        logger.info("Setting up scene...")
        self.scene_path = scene_path
        self.scene_img_path = os.path.join(scene_path, "images")
        self.scene_depth_path = os.path.join(scene_path, "depth")
        
        self.image_files = sorted(glob.glob(os.path.join(self.scene_img_path, "*.jpg")), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.depth_files = [image_file.replace("images", "depth") for image_file in self.image_files]
        
        # downsample
        self.image_files = sorted(set(self.image_files[::downsample] + [self.image_files[-1]]))
        self.depth_files = sorted(set(self.depth_files[::downsample] + [self.depth_files[-1]]))
        
        self.images, self.img_ori = self.load_initial_images(self.image_files)

        self.run_dust3r(input_images=self.images)
    
    def __call__(self, src_scene:str, stylized_path:str, output_path="output", downsample=4):
        logger.info("Running StyleLifter...")
        self.setup_scene(src_scene, downsample)
        os.makedirs(f"{output_path}/original", exist_ok=True)
        os.makedirs(f"{output_path}/stylized/", exist_ok=True)
        os.makedirs(f"{output_path}/warped", exist_ok=True)
        
        Image.open(stylized_path).convert("RGB").resize((1024, 576)).save(f"{output_path}/stylized/0.png")
        
        pil_to_tensor = transforms.ToTensor() 
        stylized_img = pil_to_tensor(Image.open(stylized_path).convert("RGB").resize((1024, 1024))).unsqueeze(0)
        _, _, H, W = self.images[0]['img'].shape
        
        # prepare the lifting
        scene_pc_list = torch.stack(self.scene.get_pts3d(clip_thred=1.)).permute(0, 3, 1, 2).detach()
        scene_pc_list = F.interpolate(scene_pc_list, size=(1024, 1024), mode='bilinear')
        src_pc = scene_pc_list[0]
        
        c2ws = self.scene.get_im_poses().detach()
        intrinsics = self.scene.get_intrinsics().detach()
        
        stylized_pc = [src_pc]
        stylized_views = [stylized_img]
        
        logger.info(f"Autoregressive style lifting of {len(self.image_files)} views")
        for tgt_id in range(1, len(self.image_files)):
            tgt_K = intrinsics[[tgt_id]]
            tgt_c2w = c2ws[tgt_id]
            tgt_depth = Image.open(self.depth_files[tgt_id]).convert("RGB")
            
            tgt_K_scaled = compute_scaled_intrinsics(tgt_K, (H, W))

            input_pc = torch.stack(stylized_pc)[[-1]]
            input_views = torch.cat(stylized_views)[[-1]]
            
            warped_img, warped_mask = project_by_splatting(
                src_img=input_views,
                src_pc=input_pc,
                tgt_pose=tgt_c2w.inverse(),
                k=tgt_K_scaled
            )
            
            warped_img.save(f"{output_path}/warped/{tgt_id}.png")
            tgt_stylized = self.ViewTransformer(
                warped_img,
                tgt_depth,
                warped_mask,
                prompt=f"A high qualitive photo of a {src_scene.split('/')[-2]}",
                guidance_scale=5.,
                num_inference_steps=100,
            ).images[0]
            
            
            tgt_stylized.resize((1024, 576)).save(f"{output_path}/stylized/{tgt_id}.png")
            
            stylized_pc.append(scene_pc_list[tgt_id])
            stylized_views.append(pil_to_tensor(tgt_stylized)[None])
        
        
        for id, img_file in enumerate(self.image_files): 
            Image.open(img_file).resize((1024, 576)).save(f"{output_path}/original/{id}.png")
        
        # Save stylized point cloud
        self._save_colored_pointcloud(stylized_pc, stylized_views, output_path)

    
    def _save_colored_pointcloud(self, stylized_pc: List[torch.Tensor],
                                stylized_views: List[torch.Tensor],
                                output_path: str,
                                filename: str = "stylized_pc.ply"):
        """
        Save a colored point cloud to .ply using stylized point clouds and RGB views.

        Args:
            stylized_pc (List[Tensor]): List of [3, H, W] tensors for each view.
            stylized_views (List[Tensor]): List of [1, 3, H, W] tensors (RGB in [0,1]).
            output_path (str): Directory to save the .ply.
            filename (str): Name of the output file (default: stylized_pc.ply).
        """
        logger.info("Saving colored stylized point cloud to PLY...")

        # Stack and reshape point positions
        pc_tensor = torch.stack(stylized_pc)                     # [N, 3, H, W]
        pc_tensor = pc_tensor.permute(0, 2, 3, 1).reshape(-1, 3)  # [N*H*W, 3]

        # Stack and reshape color data
        color_tensor = torch.cat(stylized_views)                 # [N, 3, H, W]
        color_tensor = color_tensor.permute(0, 2, 3, 1).reshape(-1, 3).clamp(0, 1)  # [N*H*W, 3]

        # Convert to numpy
        pc_np = pc_tensor.detach().cpu().numpy()
        color_np = color_tensor.detach().cpu().numpy()

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)
        pcd.colors = o3d.utility.Vector3dVector(color_np)

        # Write to file
        os.makedirs(output_path, exist_ok=True)
        ply_path = os.path.join(output_path, filename)
        o3d.io.write_point_cloud(ply_path, pcd)
        logger.info(f"Saved colored point cloud to: {ply_path}")

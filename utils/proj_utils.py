import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from splatting import splatting_function
from typing import Annotated, Tuple

def compute_scaled_intrinsics(K, orig_hw):
    H, W = orig_hw
    scale_matrix = torch.tensor([
        [1024 / W, 0, 0],
        [0, 1024 / H, 0],
        [0, 0, 1]
    ]).float().to(K.device)
    return scale_matrix @ K

def project_points(
    points_world: Annotated[torch.Tensor, "B 3 H W"],  # Points in world frame
    camera_pose: Annotated[torch.Tensor, "B 4 4"],     # World to camera transform
    K: Annotated[torch.Tensor, "B 3 3"]                  # Camera intrinsics
) -> Tuple[Annotated[torch.Tensor, "B 2 H W"],         # Projected points
           Annotated[torch.Tensor, "B 1 H W"]]:        # Depth values
    """
    Project 3D points from world coordinates to image coordinates.
    
    Args:
        points_world: World coordinates (B, 3, H, W)
        camera_pose: World to camera transformation matrix (B, 4, 4)
        K: Camera intrinsic matrix (3, 3)
    
    Returns:
        tuple: (
            projected_points: Image coordinates (B, 2, H, W),
            depths: Depth values in camera frame (B, 1, H, W)
        )
    """
    B, _, H, W = points_world.shape
    device = points_world.device
    
    # Reshape points to (B, 3, N) where N = H*W
    points = points_world.reshape(B, 3, -1)
    
    # Homogeneous coordinates
    ones = torch.ones((B, 1, H*W), device=device)
    points_h = torch.cat([points, ones], dim=1)  # (B, 4, N)
    
    # Transform points to camera frame
    points_cam = camera_pose @ points_h  # (B, 4, N)
    
    # Extract xyz coordinates and depths
    points_cam_xyz = points_cam[:, :3]  # (B, 3, N)
    depths = points_cam[:, 2:3]  # (B, 1, N)
    
    # Project to image plane
    # Divide by Z to get normalized coordinates
    points_projected = points_cam_xyz / (depths + 1e-8)  # (B, 3, N)
    
    # Apply camera intrinsics
    K = K.to(device)
    points_pixels = torch.bmm(K.expand(B, -1, -1), points_projected)  # (B, 3, N)
    
    # Convert to pixel coordinates (u, v)
    pixels_xy = points_pixels[:, :2]  # (B, 2, N)
    
    # Reshape back to original height and width
    pixels_xy = pixels_xy.reshape(B, 2, H, W)
    depths = depths.reshape(B, 1, H, W)
    
    return pixels_xy, depths

def project_by_splatting(
        src_img: torch.Tensor, 
        src_pc: torch.Tensor, 
        tgt_pose: torch.Tensor,
        k: torch.Tensor,
    ):
    """splatting the pixel values to a second view

    Args:
        src_img (torch.Tensor): B, 3, H, W
        src_pc (torch.Tensor): B, 3, H, W
        tgt_pose (torch.Tensor): world-to-frame, w2c
        k (torch.Tensor): camera intrinsic of the tgt frame
    """
    B, _, H, W = src_img.shape
    src_img = src_img.to(src_pc.device)
    coords_new, new_z = project_points(src_pc, tgt_pose, k)
    
    
    screen = torch.stack(torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing='xy'), dim=-1).to(src_pc.device, dtype=src_pc.dtype)
    
    # Masking invalid pixels.
    invalid_1 = new_z <= 0
    coords_new[invalid_1.expand(-1,2,-1,-1)] = -1000000 if coords_new.dtype == torch.float32 else -1e+4

    # Calculate flow and importance for splatting.
    forward_flow = coords_new - screen.permute(2, 0, 1)[None]
    
    ## Importance.
    importance = 0.5 / new_z
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10
    
    # splatting based mask
    img_warped = splatting_function('softmax', src_img, forward_flow, importance, eps=1e-6)
    mask = torch.any(img_warped == 0, dim=1)
    img_warped = img_warped * 255
    
    img_warped, _ = merge_warped_images(img_warped, mask)
    mask=mask.all(dim=0)  
        
    img_warped = Image.fromarray(img_warped.squeeze().permute(1,2,0).detach().cpu().numpy().astype(np.uint8)).convert("RGB")
    mask = Image.fromarray(mask.squeeze().cpu().numpy()).convert("RGB")
    
    return img_warped, mask


def merge_warped_images(img_warped_batch, mask_batch):
    """
    Merge multiple warped images while considering empty regions.
    
    Args:
        img_warped_batch (torch.Tensor): Batch of warped images [B, C, H, W]
        mask_batch (torch.Tensor): Batch of masks indicating empty regions [B, H, W]
                                 True indicates empty regions
    
    Returns:
        tuple: (merged_image, merged_mask)
            - merged_image (torch.Tensor): Merged image [C, H, W]
            - merged_mask (torch.Tensor): Combined mask [H, W], True where all images have holes
    """
    B, C, H, W = img_warped_batch.shape
    
    # Create validity weights (inverse of mask)
    validity = (~mask_batch).float()  # [B, H, W]
    
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    
    # Expand validity to match image channels
    validity = validity.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, H, W]
    
    # Weighted sum of images
    weighted_sum = (img_warped_batch * validity).sum(dim=0)  # [C, H, W]
    weight_sum = validity.sum(dim=0) + eps  # [C, H, W]
    
    # Normalize by sum of weights
    merged_image = weighted_sum / weight_sum
    
    # Compute merged mask (True where all original masks were True)
    merged_mask = mask_batch.all(dim=0)  # [H, W]
    
    return merged_image, merged_mask



def merge_warped_images_sequential(img_warped_batch, mask_batch):
    """
    Merge images sequentially - take the first valid pixel value encountered.
    
    Args:
        img_warped_batch (torch.Tensor): Batch of warped images [B, C, H, W]
        mask_batch (torch.Tensor): Batch of masks indicating empty regions [B, H, W]
                                 True indicates empty regions
    
    Returns:
        tuple: (merged_image, merged_mask)
            - merged_image (torch.Tensor): Merged image [C, H, W]
            - merged_mask (torch.Tensor): Mask showing where we have valid pixels [H, W]
    """
    B, C, H, W = img_warped_batch.shape
    
    # Initialize merged image and mask
    merged_image = torch.zeros((C, H, W), device=img_warped_batch.device)
    merged_mask = torch.zeros((H, W), device=mask_batch.device)
    
    # Create validity (inverse of mask)
    validity = ~mask_batch  # [B, H, W]
    
    # Loop through the batch sequentially
    for b in range(B):
        # Get current validity map
        current_valid = validity[b]  # [H, W]
        
        # Update only pixels that haven't been set yet (merged_mask == 0)
        # and are valid in current image (current_valid == True)
        update_pixels = current_valid & (merged_mask == 0)
        
        # Update image where needed
        merged_image[:, update_pixels] = img_warped_batch[b, :, update_pixels]
        
        # Update mask
        merged_mask[update_pixels] = 1
    
    return merged_image, merged_mask


def merge_warped_images_sequential_smooth(img_warped_batch, mask_batch):
    """
    Merge images sequentially with smooth blending at boundaries.
    
    Args:
        img_warped_batch (torch.Tensor): Batch of warped images [B, C, H, W]
        mask_batch (torch.Tensor): Batch of masks indicating empty regions [B, H, W]
                                 True indicates empty regions
    
    Returns:
        tuple: (merged_image, merged_mask)
            - merged_image (torch.Tensor): Merged image [C, H, W]
            - merged_mask (torch.Tensor): Mask showing where we have valid pixels [H, W]
    """
    B, C, H, W = img_warped_batch.shape
    
    # Initialize merged image and mask
    merged_image = torch.zeros((C, H, W), device=img_warped_batch.device)
    merged_mask = torch.zeros((H, W), device=mask_batch.device)
    
    # Initialize weight accumulator
    accumulated_weight = torch.zeros((H, W), device=mask_batch.device)
    
    # Create validity (inverse of mask)
    validity = ~mask_batch  # [B, H, W]
    
    blend_width = 5  # Width of blending region
    eps = 1e-6
    
    for b in range(B):
        current_valid = validity[b]  # [H, W]
        
        # Compute blending weight for smooth transition
        # More weight to pixels that haven't been filled yet
        blend_weight = torch.ones_like(merged_mask)
        blend_weight[accumulated_weight > 0] = torch.exp(-accumulated_weight[accumulated_weight > 0])
        
        # Update only valid pixels in current image
        update_pixels = current_valid
        
        # Apply blending
        weight = blend_weight * current_valid.float()
        weight_expanded = weight.unsqueeze(0).expand(C, -1, -1)
        
        # Update image with weighted contribution
        merged_image[:, update_pixels] = (
            merged_image[:, update_pixels] * accumulated_weight[update_pixels].unsqueeze(0) + 
            img_warped_batch[b, :, update_pixels] * weight[update_pixels].unsqueeze(0)
        ) / (accumulated_weight[update_pixels] + weight[update_pixels] + eps).unsqueeze(0)
        
        # Update accumulated weight
        accumulated_weight[update_pixels] += weight[update_pixels]
        
        # Update mask where we have any contribution
        merged_mask[update_pixels] = 1
    
    return merged_image, merged_mask
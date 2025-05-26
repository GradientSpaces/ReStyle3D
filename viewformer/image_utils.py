import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

def load_and_resize_image(path, size=(1024, 1024)):
    img = Image.open(path).convert("RGB").resize(size)
    return transforms.ToTensor()(img).unsqueeze(0)


def match_histograms_masked_full(source_img, reference_img, mask):
    """
    Match histograms based on masked region but apply to whole image
    
    Parameters:
    source_img: numpy array (H x W x 3) - Source image to be modified
    reference_img: numpy array (H x W x 3) - Reference image to match
    mask: numpy array (H x W x 3) - RGB mask
    """
    # Convert to float32
    source_float = source_img.astype(np.float32) / 255.0
    reference_float = reference_img.astype(np.float32) / 255.0
    
    # Initialize output image
    matched = source_float.copy()
    
    # Use first channel of RGB mask and ensure it's binary
    mask_channel = mask[:,:,0] if len(mask.shape) == 3 else mask
    if mask_channel.dtype != np.uint8:
        mask_channel = mask_channel.astype(np.uint8) * 255
    _, mask_binary = cv2.threshold(mask_channel, 127, 255, cv2.THRESH_BINARY)
    mask_binary = cv2.bitwise_not(mask_binary)  # Invert the mask
    
    # Create boolean mask
    bool_mask = mask_binary > 0
    
    for i in range(3):
        # Get masked pixels for computing transformation
        source_channel = source_float[:,:,i]
        reference_channel = reference_float[:,:,i]
        
        # Apply boolean mask correctly
        source_masked = source_channel[bool_mask]
        reference_masked = reference_channel[bool_mask]
        
        if len(source_masked) > 0 and len(reference_masked) > 0:
            # Use more bins for better precision
            nbins = 256
            source_hist, bin_edges = np.histogram(source_masked, nbins, [0, 1])
            reference_hist, _ = np.histogram(reference_masked, nbins, [0, 1])
            
            # Add small epsilon to avoid division by zero
            source_hist = source_hist + 1e-8
            reference_hist = reference_hist + 1e-8
            
            # Calculate normalized cumulative histograms
            source_cdf = source_hist.cumsum() / source_hist.sum()
            reference_cdf = reference_hist.cumsum() / reference_hist.sum()
            
            # Create interpolation function
            bins = np.linspace(0, 1, nbins)
            lookup_table = np.interp(source_cdf, reference_cdf, bins)
            
            # Apply transformation to entire channel
            channel_values = source_float[:,:,i] * (nbins-1)
            channel_indices = channel_values.astype(int)
            matched[:,:,i] = lookup_table[channel_indices]
    
    # Ensure output is in valid range
    matched = np.clip(matched, 0, 1)
    matched = (matched * 255).astype(np.uint8)
    
    return Image.fromarray(matched)
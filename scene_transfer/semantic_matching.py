import torch
from utils.logging import logger

def get_mask_for_label(label_name, objects_list, mask_tensor):
    """
    Get the mask for a given label from a 2D mask tensor, including all instances of the label.
    
    Args:
    label_name (str): The label to find the mask for
    objects_list (list): List of dictionaries containing object information
    mask_tensor (torch.Tensor): 2D PyTorch tensor where pixel values correspond to object IDs
    
    Returns:
    torch.Tensor: Binary mask for the given label, including all instances
    """
    # Find all objects with the given label
    target_objects = [obj for obj in objects_list if obj['label'] == label_name]
    
    if not target_objects:
        raise ValueError(f"No object found with label '{label_name}'")
    
    # Get the IDs of all target objects
    target_ids = [obj['id'] for obj in target_objects]
    
    # Create a binary mask where any of the target objects are True and everything else is False
    binary_mask = torch.zeros_like(mask_tensor, dtype=torch.bool)
    for target_id in target_ids:
        binary_mask |= (mask_tensor == target_id)
    
    return binary_mask


def match_semantic_labels(src_dict, tgt_dict):
    """
    Match semantic labels between source and target images in seg_dict.
    
    Args:
    seg_dict (dict): Dictionary containing segmentation predictions for source and target images.
    
    Returns:
    list: List of tuples containing matched labels and their similarity scores.
    """
    # Extract labels from source and target predictions
    source_labels = [obj['label'] for obj in src_dict['pred_tgt'][1]]
    target_labels = [obj['label'] for obj in tgt_dict['pred_tgt'][1]]
    # Find the intersection between source and target labels
    common_labels = list(set(source_labels) & set(target_labels))
    
    # Initialize a list to store matched labels and their masks
    matched_labels = []
    
    # Iterate through common labels
    for label in common_labels:
        # Get masks for the current label in both source and target
        source_mask = get_mask_for_label(label, src_dict['pred_tgt'][1], src_dict['pred_tgt'][0])
        target_mask = get_mask_for_label(label, tgt_dict['pred_tgt'][1], tgt_dict['pred_tgt'][0])
        
        # get the area of the source and target masks
        source_mask_area = source_mask.sum()
        target_mask_area = target_mask.sum()
        # get the area of the original images for both source and target
        source_img_area = src_dict['pred_tgt'][0].shape[-1] * src_dict['pred_tgt'][0].shape[-2]
        target_img_area = tgt_dict['pred_tgt'][0].shape[-1] * tgt_dict['pred_tgt'][0].shape[-2]
        # get the ratio of the source and target masks to the original images
        source_mask_ratio = source_mask_area / source_img_area
        target_mask_ratio = target_mask_area / target_img_area
        
        # Skip labels with mask ratios below 1%
        if source_mask_ratio < 0.01 or target_mask_ratio < 0.01:
            logger.info(f"[Semantic Matching] Skipping {label} mask due to small mask ratio")
            continue
        
        matched_labels.append((label, source_mask, target_mask))
    
    return matched_labels

def merge_similar_labels(seg_dict, labels=['wall', 'floor']):
    """
    Merge similar labels in the objects list.
    
    Args:
    objects_list (list): List of dictionaries containing object information
    
    Returns:
    list: List of dictionaries containing merged object information
    """
    for obj in seg_dict['pred_src'][1]:
        for label in labels:
            if label in obj['label']:
                obj['label'] = label
                seg_dict['pred_src'][1][seg_dict['pred_src'][1].index(obj)]['label'] = label
    
    for obj in seg_dict['pred_tgt'][1]:
        for label in labels:
            if label in obj['label']:
                obj['label'] = label
                seg_dict['pred_tgt'][1][seg_dict['pred_tgt'][1].index(obj)]['label'] = label
    
    return seg_dict


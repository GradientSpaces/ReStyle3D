import os
from pathlib import Path
import argparse

from restyle_image import generate_single_view_stylized
from viewformer.stylelifter import StyleLifter
from utils.logging import logger

def restyle_scene(scene_path: str, style_path: str, scene_type: str, output_root: str, downsample: int):
    scene_path = Path(scene_path)
    style_path = Path(style_path)
    scene_id = scene_path.parts[-1]
    style_id = style_path.parts[-1]

    # Get structure image
    image_files = sorted((scene_path / "images").glob("*.jpg"))
    if not image_files:
        logger.error(f"No images found in {scene_path / 'images'}")
        return
    struct_img = image_files[0]
    frame_name = struct_img.stem

    struct_seg = scene_path / "seg_dict" / f"{frame_name}.pth"

    # Look for style image.* (jpg or png)
    image_candidates = list(style_path.glob("image.*"))
    if not image_candidates:
        logger.error(f"No image.* found in {style_path}")
        return
    style_img = image_candidates[0]

    style_seg = style_path / "seg_dict.pth"

    # Check required files
    for p in [struct_img, struct_seg, style_img, style_seg]:
        if not p.exists():
            logger.error(f"Missing required input: {p}")
            return

    # Step 1: generate single-view stylized image
    logger.info(f"Generating single-view stylization: {style_id} → {scene_path}")
    stylized_2d_output = Path("output/2d_results") / f"{scene_id}_style_{style_id}" 
    generate_single_view_stylized(
        struct_img_path=struct_img,
        style_img_path=style_img,
        struct_seg_dict=struct_seg,
        style_seg_dict=style_seg,
        output_path=stylized_2d_output / "intermediate",
        scene_type=scene_type,
    )

    # Step 2: multi-view lifting
    logger.info(f"Starting multi-view style lifting...")
    stylelifter = StyleLifter(ckpt_path="checkpoints")
    output_3d_path = Path(output_root) / f"{scene_type}_{scene_id}" / style_id

    stylelifter(
        src_scene=str(scene_path),
        stylized_path=stylized_2d_output / "stylized.png",
        output_path=output_3d_path,
        downsample=downsample
    )

    logger.info(f"✅ Scene stylization complete for {scene_type}/{scene_id} using {style_id}.")
    logger.info(f"🖼️  Results saved to: {output_3d_path.resolve()}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReStyle3D: Scene Stylization Pipeline")
    parser.add_argument("--scene_path", type=str, required=True, help="Path to scene directory (e.g., data/interiors/bedroom/0)")
    parser.add_argument("--style_path", type=str, required=True, help="Path to style folder (e.g., data/design_styles_v2/bedroom/pexels-xxx)")
    parser.add_argument("--scene_type", type=str, required=True, help="Scene type (e.g., bedroom, kitchen, living_room)")
    parser.add_argument("--output_root", type=str, default="output/demo_restyle3d", help="Root path to save results")
    parser.add_argument("--downsample", type=int, default=4, help="Downsampling stride for multi-view processing (default: 4)")

    args = parser.parse_args()
    restyle_scene(args.scene_path, args.style_path, args.scene_type, args.output_root, args.downsample)


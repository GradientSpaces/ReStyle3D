import torch
import cv2
import diffusers


from scene_transfer.config import RunConfig
from third_party.depth_anything_v2.dpt import DepthAnythingV2


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_DepthAnyThing_model(encoder='vitl'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model

def normalize_depthmap(depthmap):
    min_, max_ = depthmap.min(), depthmap.max()
    depthmap = (depthmap - min_) / (max_ - min_) * 255
    return depthmap

def get_depthmaps(cfg: RunConfig, model='Depth-Anything'):
    if model == "Depth-Anything":
        model = get_DepthAnyThing_model()
        app_image = cv2.imread(cfg.app_image_path)
        struct_image = cv2.imread(cfg.struct_image_path)
        
        app_depth = model.infer_image(app_image)
        struct_depth = model.infer_image(struct_image)
        
    elif model == "Marigold":
        pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
        ).to(DEVICE)
        app_image = diffusers.utils.load_image(str(cfg.app_image_path))
        struct_image = diffusers.utils.load_image(str(cfg.app_image_path))
        app_depth = pipe(app_image)[0].squeeze()
        struct_depth = pipe(struct_image)[0].squeeze()
    else:
        raise NotImplementedError("unknown depth estimator!")

    app_depth = normalize_depthmap(app_depth)
    struct_depth = normalize_depthmap(struct_depth)
    
    return app_depth, struct_depth

def get_depthmap(image, model='Depth-Anything', normalize=True):
    if model == "Depth-Anything":
        model = get_DepthAnyThing_model()
        
        depth = model.infer_image(image)
        
    else:
        raise NotImplementedError("unknown depth estimator!")

    if normalize:
        depth = normalize_depthmap(depth)
    
    return depth
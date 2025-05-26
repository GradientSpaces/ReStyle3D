# 🎨 ReStyle3D: Scene-Level Appearance Transfer with Semantic Correspondences

### ACM SIGGRAPH 2025

 [![ProjectPage](https://img.shields.io/badge/Project_Page-ReStyle3D-blue)](https://restyle3d.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2502.10377-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2502.10377) [![Hugging Face (LCM) Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/gradient-spaces/ReStyle3D) [![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

Official implementation of the paper titled "Scene-level Appearance Transfer with Semantic Correspondences".

[Liyuan Zhu](http://www.kebingxin.com/)<sup>1</sup>,
[Shengqu Cai](https://primecai.github.io/)<sup>1,\*</sup>,
[Shengyu Huang](https://shengyuh.github.io/)<sup>2,\*</sup>,
[Gordon Wetzstein](https://stanford.edu/~gordonwz/)<sup>1</sup>,
[Naji Khosravan](https://www.najikhosravan.com/)<sup>3</sup>,
[Iro Armeni](https://ir0.github.io/)<sup>1</sup>



<sup>1</sup>Stanford University, <sup>2</sup>NVIDIA Research, <sup>3</sup>Zillow Group | <sup>\*</sup> denotes equal contribution


```bibtex
@inproceedings{zhu2025_restyle3d,
    author = {Liyuan Zhu and Shengqu Cai and Shengyu Huang and Gordon Wetzstein and Naji Khosravan and Iro Armeni},
    title = {Scene-level Appearance Transfer with Semantic Correspondences},
    booktitle = {ACM SIGGRAPH 2025 Conference Papers},
    year = {2025},
  }
```

We introduce ReStyle3D, a novel framework for scene-level appearance
transfer from a single style image to a real-world scene represented by
multiple views. This method combines explicit semantic correspondences
with multi-view consistency to achieve precise and coherent stylization.
<p align="center">
  <a href="">
    <img src="https://arxiv.org/html/2502.10377v1/x1.png" width="100%">
  </a>
</p>


## 🛠️ Setup
### ✅ Tested Environments
- Ubuntu 22.04 LTS, Python 3.10.15, CUDA 12.2, GeForce RTX 4090/3090

- CentOS Linux 7, Python 3.12.1, CUDA 12.4, NVIDIA A100

### 📦 Repository
```
git clone git@github.com:GradientSpaces/ReStyle3D.git
cd ReStyle3D
```

### 💻 Installation 
```
conda create -n restyle3d python=3.10
conda activate restyle3d
pip install -r requirements.txt
```

### 📦 Pretrained Checkpoints
Download the pretrained models by running:
```
bash scripts/download_weights.sh
```


## 🚀 Usage

We prepare demo data for you:
```
bash scripts/download_demo.sh
```

### 🎮 Demo (Single-view)
We include 3 demo images to run semantic appearance transfer:
```
python restyle_image.py
```



### 🎨 Stylizing Multi-view Scenes 
To run on a single scene and style:
```
python restyle_scene.py   \
 --scene_path demo/scene_transfer/bedroom/  \
 --scene_type bedroom   \
 --style_path demo/design_styles/bedroom/pexels-itsterrymag-2631746
```

### 📂 Dataset: SceneTransfer
We organize the data into two components:

1. Interior Scenes:
Multi-view real-world scans with aligned images, depth, and semantic segmentations.
```
📁 data/
  └── interiors/
      ├── bedroom/
      │   ├── 0/
      │   │   ├── images/      # multi-view RGB images
      │   │   ├── depth/       # depth maps
      │   │   └── seg_dict/    # semantic segmentation dictionaries
      │   └── 1/
      │       └── ...
      ├── living_room/
      └── kitchen/
```
2. Design Styles:
Style examplars with precomputed semantic segmentation.
```
📁 data/
  └── design_styles/
      ├── bedroom/
      │   └── pexels-itsterrymag-2631746/
      │       ├── image.jpg        # style reference image
      │       ├── seg_dict.pth     # semantic segmentation dictionary 
      │       └── seg.png          # segmentation visualization
      ├── living_room/
      └── kitchen/
```





## 🚧 TODO
- [ ] Release full dataset
- [ ] Release evaluation code
- [ ] Customize dataset


## 🙏 Acknowledgement
Our codebase is built on top of the following works:
- [Cross-image-attention](https://github.com/garibida/cross-image-attention) 
- [ODISE](https://github.com/NVlabs/ODISE)
- [ViewCrafter](https://github.com/Drexubery/ViewCrafter)
- [GenWarp](https://github.com/sony/genwarp)
- [DUSt3R](https://github.com/naver/dust3r) 

We appreciate the open-source efforts from the authors.

## 📫 Contact
If you encounter any issues or have questions, feel free to reach out: [Liyuan Zhu](liyzhu@stanford.edu).




<div align="center">

# Chain-of-Thought Guided Multi-Modal Object Re-Identification

### CVPR 2026

[![Paper](https://img.shields.io/badge/Paper-CVF-blue)](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.html)
[![PDF](https://img.shields.io/badge/PDF-CVPR%202026-red)](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.pdf)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Gaoya615/CoT-ReID)

**Ya Gao, Shihao Li, Zhaojun Liu, Aihua Zheng, Chenglong Li, Jin Tang**

</div>

## News

- **June 2026:** CoT-ReID is published at CVPR 2026. The paper is available on [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.html).
  
## Overview

CoT-ReID introduces chain-of-thought (CoT) reasoning into multi-modal object re-identification. It uses hierarchical reasoning text to guide visual representation learning at the early-feature, cross-modal semantic, and final decision levels, producing more robust multi-modal features in challenging scenarios.

Experiments are conducted on four multi-modal ReID benchmarks: **RGBNT100**, **MSVR310**, **WMVeID863**, and **RGBNT201**.

<img width="1082" height="840" alt="Comparison between human reasoning and CoT-ReID training" src="https://github.com/user-attachments/assets/3ddceba0-2b9c-465d-ae30-90ab5c3c57a6" />

<p align="center"><em>Figure 1. Human reasoning about object information and the logical reasoning process used during CoT-ReID training.</em></p>

<img width="1080" height="1124" alt="Comparison with previous text-guided ReID methods" src="https://github.com/user-attachments/assets/4f2dcf5b-03f8-4bd0-b3c8-a069b6e4d13f" />

<p align="center"><em>Figure 2. Previous text-guided ReID methods and our full-process CoT-guided training framework.</em></p>

<img width="1572" height="1090" alt="CoT-ReID framework" src="https://github.com/user-attachments/assets/db865cd8-1ad4-4b23-9e1b-3a0bad70bb46" />

<p align="center"><em>Figure 3. The overall CoT-ReID framework.</em></p>

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Gaoya615/CoT-ReID.git
cd CoT-ReID
```

### 2. Create the environment

A Linux machine with an NVIDIA GPU is recommended. Install a PyTorch build compatible with your CUDA driver by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
conda create -n cot-reid python=3.8 -y
conda activate cot-reid

# Example only: select the PyTorch/CUDA build appropriate for your machine.
pip install torch torchvision

pip install \
  numpy pillow opencv-python yacs timm transformers \
  ftfy regex tqdm einops scipy pandas matplotlib seaborn fvcore
```

> The repository does not currently include a locked environment file. If you use a newer `timm` release and encounter legacy import errors, install a compatible 0.x release (for example, `timm==0.6.12`).

## Data and Pretrained Models

The datasets, CoT text annotations, and pretrained weights are not redistributed in this repository. Prepare them locally using the following layout (or override the paths in the configuration files):

```text
CoT-ReID/
├── pretrained/
│   ├── ViT-B-16.pt
│   └── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── text/
│   ├── RGBNT201/
│   ├── MSVR310/
│   └── MSV863/
└── ...

/path/to/datasets/
├── RGBNT201/
├── MSVR310/
└── WMVeID863/
```

Before running the code:

1. Set `DATASETS.ROOT_DIR` in the corresponding file under `configs/` to your dataset root.
2. Set `DATASETS.TEXT_DIR` from the command line if your CoT annotations are not under `text/<dataset>/`.
3. Update the CLIP and DINOv3 paths in `modeling/make_model.py` if your pretrained files are stored elsewhere.
4. Update the DINOv3 source path at the top of `modeling/make_model.py` if required by your setup.

Example path override:

```bash
python train_net.py \
  --config_file configs/RGBNT201/cot-reid.yml \
  DATASETS.ROOT_DIR /path/to/datasets \
  DATASETS.TEXT_DIR /path/to/CoT-ReID/text/RGBNT201 \
  OUTPUT_DIR ./logs/RGBNT201/current
```

Please follow the official licenses and access policies of each dataset.

## Training

### RGBNT201

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/RGBNT201/cot-reid.yml \
  OUTPUT_DIR ./logs/RGBNT201/current
```

### MSVR310

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/MSVR310/cot-reid.yml \
  OUTPUT_DIR ./logs/MSVR310/current
```

### WMVeID863

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/MSV863/cot-reid.yml \
  OUTPUT_DIR ./logs/MSV863/current
```
```

## Evaluation

Replace the checkpoint path in the commands below with the model you want to evaluate.

### RGBNT201

```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py \
  --config_file configs/RGBNT201/cot-reid.yml \
  --weight logs/RGBNT201/teacher_best.pth \
  OUTPUT_DIR ./logs/RGBNT201/test
```

### MSVR310

```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py \
  --config_file configs/MSVR310/cot-reid.yml \
  --weight logs/MSVR310/teacher_best.pth \
  OUTPUT_DIR ./logs/MSVR310/test
```

### WMVeID863

```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py \
  --config_file configs/MSV863/cot-reid.yml \
  --weight logs/MSV863/teacher_best.pth \
  OUTPUT_DIR ./logs/MSV863/test
```

Additional training and evaluation notes are available in [`TRAINING.md`](TRAINING.md).

## Citation

If you find this work useful in your research, please cite:

```bibtex
@InProceedings{Gao_2026_CVPR,
  author    = {Gao, Ya and Li, Shihao and Liu, Zhaojun and Zheng, Aihua and Li, Chenglong and Tang, Jin},
  title     = {Chain-of-Thought Guided Multi-Modal Object Re-Identification},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {37705--37714}
}
```

## Acknowledgements

We thank the authors and maintainers of the datasets and open-source projects used in this work.

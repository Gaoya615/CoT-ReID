# CoT-ReID three-dataset training bundle

This folder contains the long-text training route for RGBNT201, MSVR310, and
MSV863/WMVEID863, their currently selected text annotations, DINOv3 source,
and the required CLIP/DINOv3 pretrained weights.

Image datasets are not duplicated. They remain under `/data/datasets`, as
configured by the YAML files.

Activate the training environment first:

```bash
source /data/miniconda3/etc/profile.d/conda.sh
conda activate gaoya
cd /data/gaoya/cot-reid
```

## RGBNT201

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/RGBNT201/cot-reid.yml \
  OUTPUT_DIR ./logs/RGBNT201/current
```

## MSVR310

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/MSVR310/cot-reid.yml \
  OUTPUT_DIR ./logs/MSVR310/current
```

## WMVEID863

```bash
CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --config_file configs/MSV863/cot-reid.yml \
  OUTPUT_DIR ./logs/MSV863/current
```

The default text paths are `text/RGBNT201`, `text/MSVR310`, and `text/MSV863`.
`DATASETS.TEXT_DIR` can still override a text directory explicitly.

## Test the current weights

Run the following commands from `/data/gaoya/cot-reid` after activating the
`gaoya` environment. Change `CUDA_VISIBLE_DEVICES=0` if another GPU should be
used. The evaluation logs are written to the corresponding `test_*` folders.

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
  OUTPUT_DIR ./logs/MSVR310/test_current_best
```

### WMVEID863 

```bash
CUDA_VISIBLE_DEVICES=0 python test_net.py \
  --config_file configs/MSV863/cot-reid.yml \
  --weight logs/MSV863/teacher_best.pth \
  OUTPUT_DIR ./logs/MSV863/test_current_best
```

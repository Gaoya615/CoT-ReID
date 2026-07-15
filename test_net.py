"""Evaluation-only entry point for the long-text CoT-ReID model."""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch

from config import cfg
from data.datasets.make_dataloader import make_dataloader
from engine.processor import do_inference
from modeling.make_model import make_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Evaluate long-text CoT-ReID")
    parser.add_argument("--config_file", default="configs/RGBNT201/cot-reid.yml")
    parser.add_argument("--weight", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("CoTReID", cfg.OUTPUT_DIR, if_train=False)
    logger.info("Evaluation config:\n%s", cfg)
    logger.info("Loading checkpoint: %s", args.weight)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model but is not available")

    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(args.weight, map_location="cpu")
    do_inference(cfg, model, val_loader, num_query)


if __name__ == "__main__":
    main()

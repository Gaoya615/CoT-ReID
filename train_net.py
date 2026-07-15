import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
from utils.logger import setup_logger
from data.datasets.make_dataloader import make_dataloader
from modeling.make_model import make_model
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from layers.make_loss import make_loss
from engine.processor import do_train_teacher
import random

import numpy as np
import os
import argparse
from config import cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLIP_backbone Training")
    parser.add_argument(
        "--config_file", default="configs/RGBNT201/cot-reid.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--resume_weight", default="", type=str,
        help="initialize the model from an existing checkpoint before fine-tuning",
    )
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.FEAT = args.fea_cft
    cfg.freeze()
    
    
    
    set_seed(cfg.SOLVER.SEED)
    
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(int(cfg.MODEL.DEVICE_ID))

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("CoTReID", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print("data is ready")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    if args.resume_weight:
        if not os.path.isfile(args.resume_weight):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_weight}")
        logger.info("Initializing fine-tuning from: %s", args.resume_weight)
        model.load_param(args.resume_weight, map_location="cpu")

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    do_train_teacher(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )

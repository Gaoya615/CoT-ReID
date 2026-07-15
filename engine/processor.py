import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
from config import cfg
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

def get_modality_gradient_stats(model, is_distributed=False):
    if is_distributed:
        model = model.module
    
    grad_stats = {'rgb': [], 'ni': [], 'ti': []}
    
    if hasattr(model, 'RGB_BACKBONE'):
        for param in model.RGB_BACKBONE.parameters():
            if param.grad is not None:
                grad_stats['rgb'].append(torch.norm(param.grad, p=2).item())
    elif hasattr(model, 'rgb_backbone'):
        for param in model.rgb_backbone.parameters():
            if param.grad is not None:
                grad_stats['rgb'].append(torch.norm(param.grad, p=2).item())

    if hasattr(model, 'NI_BACKBONE'):
        for param in model.NI_BACKBONE.parameters():
            if param.grad is not None:
                grad_stats['ni'].append(torch.norm(param.grad, p=2).item())
    elif hasattr(model, 'ni_backbone'):
        for param in model.ni_backbone.parameters():
            if param.grad is not None:
                grad_stats['ni'].append(torch.norm(param.grad, p=2).item())

    if hasattr(model, 'TI_BACKBONE'):
        for param in model.TI_BACKBONE.parameters():
            if param.grad is not None:
                grad_stats['ti'].append(torch.norm(param.grad, p=2).item())
    elif hasattr(model, 'ti_backbone'):
        for param in model.ti_backbone.parameters():
            if param.grad is not None:
                grad_stats['ti'].append(torch.norm(param.grad, p=2).item())
    
    grad_mean = {
        'rgb': sum(grad_stats['rgb']) / len(grad_stats['rgb']) if grad_stats['rgb'] else 0.0,
        'ni': sum(grad_stats['ni']) / len(grad_stats['ni']) if grad_stats['ni'] else 0.0,
        'ti': sum(grad_stats['ti']) / len(grad_stats['ti']) if grad_stats['ti'] else 0.0
    }
    return grad_mean

def do_train_teacher(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"

    logger = logging.getLogger("CoTReID.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    reid_loss_meter = AverageMeter()
    cmi_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=str(cfg.TEST.RE_RANKING).lower() == 'yes',
            reranking_k1=cfg.TEST.RE_RANKING_K1,
            reranking_k2=cfg.TEST.RE_RANKING_K2,
            reranking_lambda=cfg.TEST.RE_RANKING_LAMBDA,
        )
    
    scaler = amp.GradScaler()
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}

    gradient_log = {
        'epoch': [],
        'rgb_grad': [],
        'ni_grad': [],
        'ti_grad': []
    }

    print('-'*20, '第一阶段：训练教师模型', '-'*20)
    
    for epoch in tqdm(range(1, epochs + 1)):
        start_time = time.time()
        loss_meter.reset()
        reid_loss_meter.reset()
        cmi_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        epoch_grad_mean = None

        for n_iter, (img, vid, target_cam, target_view, batch_indices, img_paths, text) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            text = {k: v.to(device) for k, v in text.items()}

            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with torch.amp.autocast('cuda', enabled=True):
                score, feat, loss_1 = model(
                    img, 
                    text,
                    label=target, 
                    cam_label=target_cam, 
                    view_label=target_view,
                    img_path=img_paths
                )
                loss_cls = loss_fn(score, feat, target, target_cam, current_step=epoch)
                loss = loss_cls + loss_1#*0.05
            scaler.scale(loss).backward()
           
            if local_rank == 0:
                is_distributed = cfg.MODEL.DIST_TRAIN and torch.cuda.device_count() > 1
                epoch_grad_mean = get_modality_gradient_stats(model, is_distributed)

            scaler.step(optimizer)
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
            scaler.update()

            with torch.no_grad():
                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            reid_loss_meter.update(loss_cls.item(), img['RGB'].shape[0])
            cmi_loss_meter.update(loss_1.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ReID: {:.3f}, CMI(w): {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, Margin: {:.3f}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, reid_loss_meter.avg, cmi_loss_meter.avg,
                                    acc_meter.avg, scheduler._get_lr(epoch)[0],
                                    getattr(loss_fn, "current_margin", cfg.SOLVER.MARGIN)))

        if local_rank == 0 and epoch_grad_mean is not None:
            gradient_log['epoch'].append(epoch)
            gradient_log['rgb_grad'].append(epoch_grad_mean['rgb'])
            gradient_log['ni_grad'].append(epoch_grad_mean['ni'])
            gradient_log['ti_grad'].append(epoch_grad_mean['ti'])

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0 and local_rank == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if not cfg.MODEL.DIST_TRAIN:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, img_paths, text) in enumerate(val_loader):
                    with torch.no_grad():
                        img = {
                            'RGB': img['RGB'].to(device),
                            'NI': img['NI'].to(device),
                            'TI': img['TI'].to(device)
                        }
                        text = {k: v.to(device) for k, v in text.items()}

                        camids = camids.to(device)
                        if isinstance(target_view, (list, tuple)):
                            target_view = torch.tensor(target_view, device=device)
                        else:
                            target_view = target_view.to(device)
                                                
                        feat = model(img, text, label=vid, cam_label=camids, view_label=target_view, img_path=img_paths)
                        
                
                        feat_cpu = feat.detach().cpu()
                        vid_cpu = vid.cpu() if torch.is_tensor(vid) else vid
                        camid_cpu = camid.cpu() if torch.is_tensor(camid) else camid
                        target_view_cpu = target_view.cpu() if torch.is_tensor(target_view) else target_view

                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator.update((feat_cpu, vid_cpu, camid_cpu, target_view_cpu, img_paths))
                        else:
                            evaluator.update((feat_cpu, vid_cpu, camid_cpu))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(), f"{cfg.OUTPUT_DIR}/teacher_best.pth")
                
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                torch.cuda.empty_cache()

    if local_rank == 0:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'last_logits': score
        }, f"{cfg.OUTPUT_DIR}/teacher_final.pth")

    if local_rank == 0 and len(gradient_log['epoch']) > 0:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rgb_smoothed = gaussian_filter1d(gradient_log['rgb_grad'], sigma=1)
        ni_smoothed = gaussian_filter1d(gradient_log['ni_grad'], sigma=1)
        ti_smoothed = gaussian_filter1d(gradient_log['ti_grad'], sigma=1)
        
        ax.plot(gradient_log['epoch'], rgb_smoothed, label='RGB Branch', color='red', linewidth=2)
        ax.plot(gradient_log['epoch'], ni_smoothed, label='NI Branch', color='blue', linewidth=2)
        ax.plot(gradient_log['epoch'], ti_smoothed, label='TI Branch', color='green', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Gradient L2 Norm')
        ax.set_title('Three-Modality Gradient Dynamics')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'gradient_curve.png'), dpi=300)
        pd.DataFrame(gradient_log).to_csv(os.path.join(cfg.OUTPUT_DIR, 'gradient_data.csv'), index=False)


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("CoTReID.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=str(cfg.TEST.RE_RANKING).lower() == 'yes',
            reranking_k1=cfg.TEST.RE_RANKING_K1,
            reranking_k2=cfg.TEST.RE_RANKING_K2,
            reranking_lambda=cfg.TEST.RE_RANKING_LAMBDA,
        )
    
    evaluator.reset()
    
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    
    for n_iter, (img, pid, camid, camids, target_view, img_paths, text) in enumerate(val_loader):
        with torch.no_grad():
            img = {
                'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            text = {k: v.to(device) for k, v in text.items()}

            camids = camids.to(device)
            target_view = target_view.to(device)
            
            feat = model(img, text, cam_label=camids, view_label=target_view)
            
            feat_cpu = feat.detach().cpu()
            pid_cpu = pid.cpu() if torch.is_tensor(pid) else pid
            camid_cpu = camid.cpu() if torch.is_tensor(camid) else camid
            target_view_cpu = target_view.cpu() if torch.is_tensor(target_view) else target_view

            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat_cpu, pid_cpu, camid_cpu, target_view_cpu, img_paths))
            else:
                evaluator.update((feat_cpu, pid_cpu, camid_cpu))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Test Results:")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1,5,10]:
        logger.info("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    return cmc[0], cmc[4]

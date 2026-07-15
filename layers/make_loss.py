# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
#from .triplet_loss import TripletLoss

from .center_loss import CenterLoss

import os
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            if cfg.SOLVER.DYNAMIC_MARGIN:
                print(
                    "using progressive stage-1 triplet margin: "
                    "{:.3f} -> {:.3f}, ramp epochs {}-{}".format(
                        cfg.SOLVER.MARGIN_START,
                        cfg.SOLVER.MARGIN,
                        cfg.SOLVER.MARGIN_RAMP_START,
                        cfg.SOLVER.MARGIN_RAMP_END,
                    )
                )
            else:
                print("using fixed triplet margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, normalize_feature=False, current_step=0):
            solver = cfg.SOLVER
            if solver.DYNAMIC_MARGIN:
                if current_step <= solver.MARGIN_RAMP_START:
                    current_margin = solver.MARGIN_START
                elif current_step >= solver.MARGIN_RAMP_END:
                    current_margin = solver.MARGIN
                else:
                    progress = (
                        (current_step - solver.MARGIN_RAMP_START)
                        / (solver.MARGIN_RAMP_END - solver.MARGIN_RAMP_START)
                    )
                    current_margin = solver.MARGIN_START + progress * (
                        solver.MARGIN - solver.MARGIN_START
                    )
                triplet.margin = float(current_margin)
                triplet.ranking_loss.margin = float(current_margin)
                loss_func.current_margin = float(current_margin)
            else:
                loss_func.current_margin = float(solver.MARGIN)

            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)/ len(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        #TRI_LOSS = [triplet(feats, target, normalize_feature=normalize_feature, current_step=current_step)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)    
                    else:
                        TRI_LOSS = triplet(feat, target)[0]
                        #TRI_LOSS = [triplet(feats, target, normalize_feature=normalize_feature, current_step=current_step)[0] for feats in feat[0:]]
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion








# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# def make_loss(cfg, num_classes):  # modified by gu
#     sampler = cfg.DATALOADER.SAMPLER
#     feat_dim = 2048
#     center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False)  # center loss
#     if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
#         if cfg.MODEL.NO_MARGIN:
#             triplet = TripletLoss()
#             print("using soft triplet loss for training")
#         else:
#             triplet = TripletLoss(cfg.SOLVER.MARGIN)   # triplet loss
#             print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
#     else:
#         print('expected METRIC_LOSS_TYPE should be triplet'
#               'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
# ####
#     if cfg.MODEL.IF_LABELSMOOTH == 'on':
#         xent = CrossEntropyLabelSmooth(num_classes=num_classes)
#         print("label smooth on, numclasses:", num_classes)

#     if sampler == 'softmax':#否
#         def loss_func(score, feat, target, target_cam):
#             return F.cross_entropy(score, target)
# # 在make_loss函数中集成动态TripletLoss
# # def make_loss(cfg, num_classes):
# #     sampler = cfg.DATALOADER.SAMPLER
# #     feat_dim = 2048
# #     center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=False)
    
# #     # 解析动态margin参数
# #     dynamic_margin_config = {
# #         "initial_margin": cfg.SOLVER.MARGIN,  # 阶段1初始margin
# #         "warmup_epochs": cfg.SOLVER.WARMUP_EPOCHS,   # 热身轮次
# #         "total_epochs": cfg.SOLVER.MAX_EPOCHS        # 总轮次
# #     }
    
# #     if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
# #         triplet = TripletLoss(**dynamic_margin_config)
       
# #     else:
# #         raise ValueError(f"Unsupported METRIC_LOSS_TYPE: {cfg.MODEL.METRIC_LOSS_TYPE}")
    
# #     if cfg.MODEL.IF_LABELSMOOTH == 'on':
# #         xent = CrossEntropyLabelSmooth(num_classes=num_classes)
# #         print(f"Label smooth on, num_classes: {num_classes}")
# #     else:
# #         xent = LabelSmoothingCrossEntropy(num_classes=num_classes) if cfg.MODEL.IF_LABELSMOOTH == 'lsce' else nn.CrossEntropyLoss()
    
#     if sampler == 'softmax_triplet':
#           def loss_func(score, feat, target, target_cam):
#             if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
#                 if cfg.MODEL.IF_LABELSMOOTH == 'on':
#                     if isinstance(score, list):
#                         # ID_LOSS = [xent(scor, target) for scor in score[1:]]
#                         # ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#                         if cfg.MODEL.APART == 'off':
#                             # ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
#                             # ID_LOSS_MAIN = triplet(feat[0], target)[0]
    
#                             ID_LOSS = [xent(scor, target) for scor in score[0:]]
#                             # ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#                             ID_LOSS =  sum(ID_LOSS)
#                         else:
#                             # ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
#                             ID_LOSS_MAIN = triplet(feat[0], target)[0]
    
#                             ID_LOSS = [xent(scor, target) for scor in score[1:]]
#                             ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#                             ID_LOSS =  0.5*ID_LOSS + 0.5*ID_LOSS_MAIN
#                     else:
#                         ID_LOSS = xent(score, target)

#                     if isinstance(feat, list):
#                         if cfg.MODEL.APART == 'off':
#                             # print("here is all")
#                             # TRI_LOSS_MAIN = triplet(feat[0], target)[0] 
#                             TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]#若feat是多分支特征（列表），对各分支损失取平均并加权
#                             # TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
#                             # TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]## 主分支与辅助分支损失加权平均
#                             TRI_LOSS = sum(TRI_LOSS) 
#                         else:
#                             # print("here is apart")
#                             TRI_LOSS_MAIN = triplet(feat[0], target)[0] 
#                             TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]#若feat是多分支特征（列表），对各分支损失取平均并加权
#                             TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
#                             # TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]## 主分支与辅助分支损失加权平均
#                             TRI_LOSS = 0.5*TRI_LOSS + 0.5 * TRI_LOSS_MAIN# 主分支与辅助分支损失加权平均
#                     else:
#                         TRI_LOSS = triplet(feat, target)[0]

#                     return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
#                         cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
#                 else:
#                     if isinstance(score, list):
#                         ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
#                         ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#                         ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
#                     else:
#                         ID_LOSS = F.cross_entropy(score, target)

#                     if isinstance(feat, list):
#                         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
#                         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
#                         TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
#                     else:
#                         TRI_LOSS = triplet(feat, target)[0]

#                     return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
#                         cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
#             else:
#                 print('expected METRIC_LOSS_TYPE should be triplet'
#                       'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
#     # elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':#组合使用交叉熵损失和三元组损失
#     #     def loss_func(score, feat, target, target_cam):
#     #         if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
#     #             if cfg.MODEL.IF_LABELSMOOTH == 'on':
#     #                 if isinstance(score, list):
#     #                     # ID_LOSS = [xent(scor, target) for scor in score[1:]]
#     #                     # ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#     #                     if cfg.MODEL.APART == 'off':
#     #                         # ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
#     #                         ID_LOSS_MAIN = triplet(feat[0], target)[0]
    
#     #                         ID_LOSS = [xent(scor, target) for scor in score[1:]]
#     #                         ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#     #                         ID_LOSS =  0.5 *ID_LOSS + 0.5*ID_LOSS_MAIN
#     #                     else:
#     #                         # ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
#     #                         ID_LOSS_MAIN = triplet(feat[0], target)[0]
    
#     #                         ID_LOSS = [xent(scor, target) for scor in score[1:]]
#     #                         ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#     #                         ID_LOSS =  0.5*ID_LOSS + 0.5*ID_LOSS_MAIN
#     #                 else:
#     #                     ID_LOSS = xent(score, target)

#     #                 if isinstance(feat, list):
#     #                     if cfg.MODEL.APART == 'off':
#     #                         # print("here is all")
#     #                         TRI_LOSS_MAIN = triplet(feat[0], target)[0] 
#     #                         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]#若feat是多分支特征（列表），对各分支损失取平均并加权
#     #                         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
#     #                         # TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]## 主分支与辅助分支损失加权平均
#     #                         TRI_LOSS = 0.5*TRI_LOSS + 0.5 * TRI_LOSS_MAIN# 主分支与辅助分支损失加权平均
#     #                     else:
#     #                         # print("here is apart")
#     #                         TRI_LOSS_MAIN = triplet(feat[0], target)[0] 
#     #                         TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]#若feat是多分支特征（列表），对各分支损失取平均并加权
#     #                         TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
#     #                         # TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]## 主分支与辅助分支损失加权平均
#     #                         TRI_LOSS = 0.5*TRI_LOSS + 0.5 * TRI_LOSS_MAIN# 主分支与辅助分支损失加权平均
#     #                 else:
#     #                     TRI_LOSS = triplet(feat, target)[0]

#     #                 return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
#     #                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
#     #             else:
#     #                 if isinstance(score, list):
#     #                     ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
#     #                     ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#     #                     ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
#     #                 else:
#     #                     ID_LOSS = F.cross_entropy(score, target)

#     #                 if isinstance(feat, list):
#     #                     TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
#     #                     TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
#     #                     TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
#     #                 else:
#     #                     TRI_LOSS = triplet(feat, target)[0]

#     #                 return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
#     #                     cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
#     #         else:
#     #             print('expected METRIC_LOSS_TYPE should be triplet'
#     #                   'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

#     # else:
#     #     print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
#     #           'but got {}'.format(cfg.DATALOADER.SAMPLER))
#     return loss_func, center_criterion

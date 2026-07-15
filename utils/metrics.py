import numpy as np
import torch
from utils.reranking import re_ranking


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.array([])
        if isinstance(x[0], torch.Tensor):
            return torch.stack([v.detach().cpu() if torch.is_tensor(v) else torch.tensor(v) for v in x]).numpy()
        return np.asarray(x)
    return np.asarray(x)


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(mat1=qf, mat2=gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()

def eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank: max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_sceneid = q_sceneids[q_idx]
        order = indices[q_idx]
        
        remove = (g_pids[order] == q_pid) & (g_sceneids[order] == q_sceneid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc): continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        AP = (np.asarray(tmp_cmc) * orig_cmc).sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank: max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc, all_AP, num_valid_q = [], [], 0.
    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc): continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        AP = (np.asarray(tmp_cmc) * orig_cmc).sum() / num_rel
        all_AP.append(AP)
    
    all_cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    return all_cmc, np.mean(all_AP)

class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        self.num_query, self.max_rank, self.feat_norm = num_query, max_rank, feat_norm
        self.reset()

    def reset(self):
        self.feats, self.pids, self.camids, self.sceneids, self.img_path = [], [], [], [], []

    def update(self, output):
        feat, pid, camid, sceneid, img_path = output
        self.feats.append(feat.detach().cpu())
        self.pids.extend(_to_numpy(pid))
        self.camids.extend(_to_numpy(camid))
        self.sceneids.extend(_to_numpy(sceneid))
        self.img_path.extend(img_path)

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        qf, gf = feats[:self.num_query], feats[self.num_query:]
        q_pids, q_camids, q_sceneids = np.asarray(self.pids[:self.num_query]), np.asarray(self.camids[:self.num_query]), np.asarray(self.sceneids[:self.num_query])
        g_pids, g_camids, g_sceneids = np.asarray(self.pids[self.num_query:]), np.asarray(self.camids[self.num_query:]), np.asarray(self.sceneids[self.num_query:])
        
        distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False,
                 reranking_k1=20, reranking_k2=6, reranking_lambda=0.3):
        if isinstance(feat_norm, str):
            feat_norm = feat_norm.lower() == 'yes'
        if isinstance(reranking, str):
            reranking = reranking.lower() == 'yes'
        self.num_query, self.max_rank = num_query, max_rank
        self.feat_norm, self.reranking = feat_norm, reranking
        self.reranking_k1 = int(reranking_k1)
        self.reranking_k2 = int(reranking_k2)
        self.reranking_lambda = float(reranking_lambda)
        self.reset()

    def reset(self):
        self.feats, self.pids, self.camids = [], [], []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat.detach().cpu())
        self.pids.extend(_to_numpy(pid))
        self.camids.extend(_to_numpy(camid))
        
    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        qf, gf = feats[:self.num_query], feats[self.num_query:]
        q_pids, q_camids = np.asarray(self.pids[:self.num_query]), np.asarray(self.camids[:self.num_query])
        g_pids, g_camids = np.asarray(self.pids[self.num_query:]), np.asarray(self.camids[self.num_query:])

        if self.reranking:
            distmat = re_ranking(
                qf,
                gf,
                k1=self.reranking_k1,
                k2=self.reranking_k2,
                lambda_value=self.reranking_lambda,
            )
        else:
            distmat = euclidean_distance(qf, gf)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
# import torch
# import numpy as np
# import os
# from utils.reranking import re_ranking
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import manifold
# import random


# def euclidean_distance(qf, gf):
#     m = qf.shape[0]
#     n = gf.shape[0]
#     dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
#     return dist_mat.cpu().numpy()


# def cosine_similarity(qf, gf):
#     epsilon = 0.00001
#     dist_mat = qf.mm(gf.t())
#     qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
#     gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
#     qg_normdot = qf_norm.mm(gf_norm.t())

#     dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
#     dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
#     dist_mat = np.arccos(dist_mat)
#     return dist_mat


# def eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids, max_rank=50):
#     """Evaluation with market1501 metric
#         Key: for each query identity, its gallery images from the same camera view are discarded.
#         """
#     num_q, num_g = distmat.shape
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)

#     query_arg = np.argsort(q_pids, axis=0)
#     result = g_pids[indices]
#     gall_re = result[query_arg]
#     gall_re = gall_re.astype(np.str_)
#     # pdb.set_trace()

#     result = gall_re[:, :100]

#     # with open("re.txt", 'w') as file_obj:
#     #     for li in result:
#     #         for j in range(len(li)):
#     #             if j == len(li) - 1:
#     #                 file_obj.write(li[j] + "\n")
#     #             else:
#     #                 file_obj.write(li[j] + " ")
#     with open('re.txt', 'w') as f:
#         f.write('rank list file\n')

#     # pdb.set_trace()
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         q_sceneid = q_sceneids[q_idx]

#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]
#         # original protocol in RGBNT100 or RGBN300
#         # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)

#         # for each query sample, its gallery samples from same scene with same or neighbour view are discarded # added by zxp
#         # symmetrical_cam = (8 - q_camid) % 8
#         # remove = (g_pids[order] == q_pid) & ( # same id
#         #              (g_sceneids[order] == q_sceneid) & # same scene
#         #              ((g_camids[order] == q_camid) | (g_camids[order] == (q_camid + 1)%8) | (g_camids[order] == (q_camid - 1)%8) | # neighbour cam with q_cam
#         #              (g_camids[order] == symmetrical_cam) | (g_camids[order] == (symmetrical_cam + 1)%8) | (g_camids[order] == (symmetrical_cam - 1)%8)) # nerighboour cam with symmetrical cam
#         #          )
#         # new protocol in MSVR310
#         remove = (g_pids[order] == q_pid) & (g_sceneids[order] == q_sceneid)
#         keep = np.invert(remove)

#         with open('re.txt', 'a') as f:
#             f.write('{}_s{}_v{}:\n'.format(q_pid, q_sceneid, q_camid))
#             v_ids = g_pids[order][keep][:max_rank]
#             v_cams = g_camids[order][keep][:max_rank]
#             v_scenes = g_sceneids[order][keep][:max_rank]
#             for vid, vcam, vscene in zip(v_ids, v_cams, v_scenes):
#                 f.write('{}_s{}_v{}  '.format(vid, vscene, vcam))
#             f.write('\n')

#         # compute cmc curve
#         # binary vector, positions with value 1 are correct matches
#         orig_cmc = matches[q_idx][keep]
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue

#         cmc = orig_cmc.cumsum()
#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)

#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)

#     return all_cmc, mAP


# def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
#     """Evaluation with market1501 metric
#         Key: for each query identity, its gallery images from the same camera view are discarded.
#         """
#     num_q, num_g = distmat.shape
#     # distmat g
#     #    q    1 3 2 4
#     #         4 1 2 3
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     #  0 2 1 3
#     #  1 2 3 0
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]  # select one row
#         remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#         keep = np.invert(remove)

#         # compute cmc curve
#         # binary vector, positions with value 1 are correct matches
#         orig_cmc = matches[q_idx][keep]
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue

#         cmc = orig_cmc.cumsum()
#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
#         tmp_cmc = tmp_cmc / y
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)

#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)

#     return all_cmc, mAP


# class R1_mAP():
#     def __init__(self, num_query, max_rank=50, feat_norm='yes'):
#         super(R1_mAP, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []
#         self.sceneids = []
#         self.img_path = []

#     def update(self, output):
#         feat, pid, camid, sceneid, img_path = output
#         self.feats.append(feat)
#         # if feat == 'tensor':
#         #     # 输入为单个张量，直接添加
#         #     self.feats.append(feat)
#         # elif feat == 'list':
#         #     # 输入为特征列表，需要展平并记录每个特征的来源
#         #     for f in feat:
#         #         self.feats.append(f)
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))
#         self.sceneids.extend(np.asarray(sceneid))
#         self.img_path.extend(img_path)

#     def compute(self):
#         # print(len(self.feats))
#         feats = torch.cat(self.feats, dim=0)#######
#         if self.feat_norm == 'yes':
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)
#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])

#         q_sceneids = np.asarray(self.sceneids[:self.num_query])  # zxp
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])
#         g_camids = np.asarray(self.camids[self.num_query:])

#         g_sceneids = np.asarray(self.sceneids[self.num_query:])  # zxp

#         m, n = qf.shape[0], gf.shape[0]
#         distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         distmat.addmm_(1, -2, qf, gf.t())
#         distmat = distmat.cpu().numpy()
#         cmc, mAP = eval_func_msrv(distmat, q_pids, g_pids, q_camids, g_camids, q_sceneids, g_sceneids)
#         return cmc, mAP, distmat, self.pids, self.camids, qf, gf


# class R1_mAP_eval():
#     def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
#         super(R1_mAP_eval, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.reranking = reranking

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []

#     def update(self, output):  # called once for each batch
#         feat, pid, camid = output
#         self.feats.append(feat.cpu())
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))

#     def compute(self):  # called after each epoch
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm:
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])

#         g_camids = np.asarray(self.camids[self.num_query:])
#         if self.reranking:
#             print('=> Enter reranking')
#             # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
#             distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

#         else:
#             print('=> Computing DistMat with euclidean_distance')
#             distmat = euclidean_distance(qf, gf)

#         cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
#         return cmc, mAP, distmat, self.pids, self.camids, qf, gf

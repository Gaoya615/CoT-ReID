import sys
sys.path.insert(0, "/data/gaoya/cot-reid/dinov3-main")

# 接下来才是你原本的其他 import
import os
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from utils.flops import give_supported_ops
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
import torch.nn.functional as F
from fvcore.nn import flop_count
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming
import torch
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import pdb
from tqdm import tqdm
from .clip import clip
import numpy as np
import sys
import os
from dinov3.models.vision_transformer import vit_base
from dinov3.hub.backbones import Weights, _make_dinov3_vit

def load_clip_to_cpu(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size):
    model_path = '/data/gaoya/cot-reid/pretrained/ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(cfg,state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B,N, C = y.shape
        # print("!!!!!!!")
        # print(self.normy(x).shape)#（8，12，128，64）

        q = self.q_(self.normy(x)).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # 复用CLIP原始组件（补充token_embedding）
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding  # [77, 512]
        self.token_embedding = clip_model.token_embedding  # 新增：暴露token嵌入层
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.context_length = clip_model.context_length  # 固定为77
        self.tokenizer = _Tokenizer()  # 复用CLIP原始分词器

    def forward(self, text_embeddings, tokenized_prompts):
        x = text_embeddings + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x
        
from dinov3.models.vision_transformer import vit_base  # 导入base版本模型构建函数
import copy
import os
import numpy
from dinov3.hub.backbones import Weights, _make_dinov3_vit

def load_dinov3_vitb16(weights_path_or_url: str, check_hash: bool = False) -> torch.nn.Module:
    """
    加载DINOv3 ViT-Base (16x16 patch)模型及权重
    
    参数:
        weights_path_or_url: 权重文件的本地路径或URL
        check_hash: 是否验证权重文件的哈希值
    
    返回:
        加载好权重的DINOv3 ViT-Base模型
    """
    # 调用底层构建函数，指定ViT-Base配置
    model = _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,          # ViT-Base特征维度
        depth=12,               # ViT-Base层数
        num_heads=12,           # ViT-Base注意力头数
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=True,
        weights=weights_path_or_url,  # 可以是本地路径或URL
        compact_arch_name="vitb",     # 指定ViT-Base架构
        check_hash=check_hash
    )
    return model

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(Backbone, self).__init__()
 
        model_path = "/data/gaoya/cot-reid/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        # model_name = cfg.MODEL.NAME
        # pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE

        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
    
        self.base = load_dinov3_vitb16(model_path)
        # self.base.load_state_dict(self.state_dict, strict=False)

   

    def forward(self, x, cot, label=None, cam_label=None, view_label=None):
        feature = self.base(x, cot)
        return feature["x_norm_patchtokens"], feature["x_norm_clstoken"]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class TextTokenProcessor(nn.Module):
    """文本Token预处理：语义对齐+双角色适配（上下文T+独立模态）"""
    def __init__(self, text_dim=512, visual_dim=512):
        super().__init__()
        # 1. 语义对齐：确保文本与视觉特征空间一致（InfoBridge要求模态特征维度统一）
        self.feat_proj = nn.Linear(text_dim, visual_dim)  # 文本→视觉维度映射
        self.feat_norm = nn.LayerNorm(visual_dim)         # 归一化，减少模态分布偏差
        
        # 2. 上下文T提取：从文本Token中提取语义类别/场景信息（InfoBridge的T需任务相关）
        self.context_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim//2),
            nn.ReLU(),
            nn.Linear(text_dim//2, 10)  # 假设输出10个语义类别（如场景：夜晚/白天；服饰：红色/蓝色）
        )

    def forward(self, text_token):
        """
        Args:
            text_token: 文本上下文Token，shape [batch_size, text_dim] = [32, 512]
        Returns:
            text_modal: 作为独立模态的文本特征，shape [32, 512, 1, 1]（匹配视觉特征空间）
            text_context_T: 作为上下文T的语义信息，shape [32, 10]（语义类别分布）
        """
        # 角色1：独立模态特征（适配视觉模态的空间维度，如[B, C, H, W]）
        text_feat = self.feat_proj(text_token)  # [32, 512]
        text_feat = self.feat_norm(text_feat)   # [32, 512]
        text_modal = text_feat.unsqueeze(-1).unsqueeze(-1)  # [32, 512, 1, 1]
        
        # 角色2：上下文T（语义引导信息）
        text_context_T = self.context_proj(text_token)  # [32, 10]
        text_context_T = F.softmax(text_context_T, dim=1)  # 转为类别概率分布，符合InfoBridge的T定义
        
        return text_modal, text_context_T

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerModalTextProcessor(nn.Module):
    """
    单模态文本Token处理器：为每个模态的文本Token实现模态化（适配视觉空间）与语义对齐（适配视觉分布）
    """
    def __init__(self, text_dim=512, visual_dim=512, num_modals=3):
        super().__init__()
        self.num_modals = num_modals  # 模态数量（如RGB/NI/TI共3个）
        # 为每个模态的文本Token配置独立线性投影（确保模态专属语义不混淆，论文“模态特异性保留”逻辑）
        self.modal_text_proj = nn.ModuleList([
            nn.Linear(text_dim, visual_dim) for _ in range(num_modals)
        ])
        self.modal_text_norm = nn.ModuleList([
            nn.LayerNorm(visual_dim) for _ in range(num_modals)
        ])
        # 文本Token扩展空间维度（匹配视觉特征的[B, C, H, W]格式，论文视觉特征处理逻辑）
        self.spatial_expand = lambda x: x.unsqueeze(-1).unsqueeze(-1)

    def forward(self, per_modal_text_tokens):
        """
        Args:
            per_modal_text_tokens: 一模态一文本Token列表，长度=num_modals，每个元素shape [B, text_dim] = [32, 512]
        Returns:
            text_modals: 模态化后的文本特征列表，每个元素shape [32, 512, 1, 1]（匹配视觉特征空间）
            text_context_T: 上下文T列表，每个元素shape [32, 512]（模态专属语义引导信息，适配论文T定义）
        """
        text_modals = []
        text_context_T = []
        # pdb.set_trace()
        for idx in range(self.num_modals):
            # 1. 文本Token维度对齐（文本512→视觉512）
            text_proj = self.modal_text_proj[idx](per_modal_text_tokens[idx])  # [32, 512]
            # 2. 分布归一化（减少文本-视觉模态异质性，论文3.2节“模态对齐前提”）
            text_norm = self.modal_text_norm[idx](text_proj)  # [32, 512]
            # 3. 模态化：扩展空间维度（匹配视觉特征的[B, C, H, W]）
            text_modal = self.spatial_expand(text_norm)  # [32, 512, 1, 1]
            # 4. 提取模态专属上下文T（文本语义作为T，引导CMI优化，）
            text_T = text_norm  # 直接使用归一化后的文本特征作为T（语义信息完整）
            
            text_modals.append(text_modal)
            text_context_T.append(text_T)
        return text_modals, text_context_T


class CMIPriorNetwork(nn.Module):
    """Learnable conditional prior shared by all cross-modal directions."""

    def __init__(self, context_dim=512, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_head = nn.Linear(hidden_dim, 2)
        self.neg_head = nn.Linear(hidden_dim, 2)

    def forward(self, context):
        feat = self.shared(context)
        pos_params = self.pos_head(feat)
        neg_params = self.neg_head(feat)
        pos_mu = torch.sigmoid(pos_params[:, 0]) * 0.5 + 0.5
        pos_logvar = torch.clamp(pos_params[:, 1], -5, 5)
        neg_mu = torch.sigmoid(neg_params[:, 0]) * 0.5
        neg_logvar = torch.clamp(neg_params[:, 1], -5, 5)
        return (pos_mu, pos_logvar), (neg_mu, neg_logvar)


class CMICritic(nn.Module):
    """Persistent discriminator for one directed cross-modal pair."""

    def __init__(self, feat_dim=512, context_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feat_dim * 2 + context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, pair):
        return self.network(pair)


        
class CoTReID(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory, feat_dim=768):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model_name = 'ViT-B-16'
        self.text_feat_dim = 512

        # Shared hierarchical attention for description and CoT chunks.  The
        # last layer starts at zero, so training begins with the old uniform
        # pooling behaviour and learns which chunks are more informative.
        self.chunk_attention = nn.Sequential(
            nn.LayerNorm(self.text_feat_dim),
            nn.Linear(self.text_feat_dim, self.text_feat_dim // 4),
            nn.Tanh(),
            nn.Linear(self.text_feat_dim // 4, 1, bias=False),
        )
        nn.init.zeros_(self.chunk_attention[-1].weight)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1) 
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.dataset_name = cfg.DATASETS.NAMES
        clip_model = load_clip_to_cpu(
            cfg, 
            backbone_name=self.model_name, 
            h_resolution=self.h_resolution,
            w_resolution=self.w_resolution,
            vision_stride_size=self.vision_stride_size
        )
        clip_model.to(self.device)
        self.RGB_BACKBONE = Backbone(
            num_classes=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            cfg=cfg
        )
        self.NI_BACKBONE = Backbone(
            num_classes=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            cfg=cfg
        )
        self.TI_BACKBONE = Backbone(
            num_classes=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            cfg=cfg
        )
        self.text_token_proj = nn.Linear(self.text_feat_dim, self.feat_dim)
        self.fc_r = nn.Linear(self.text_feat_dim, num_classes, bias=False)
        self.fc_n = nn.Linear(self.text_feat_dim, num_classes, bias=False)
        self.fc_t = nn.Linear(self.text_feat_dim, num_classes, bias=False)
      
        self.bn_r = nn.BatchNorm1d(self.text_feat_dim)
        self.bn_n = nn.BatchNorm1d(self.text_feat_dim)
        self.bn_t = nn.BatchNorm1d(self.text_feat_dim)
       
        self.fc = nn.Linear(3 * self.text_feat_dim, num_classes, bias=False)

        self.fc_r_t = nn.Linear(self.text_feat_dim, num_classes, bias=False)
        self.fc_n_t = nn.Linear(self.text_feat_dim, num_classes, bias=False)
        self.fc_t_t = nn.Linear(self.text_feat_dim, num_classes, bias=False)

        self.bn_r_t = nn.BatchNorm1d(self.text_feat_dim)
        self.bn_n_t = nn.BatchNorm1d(self.text_feat_dim)
        self.bn_t_t = nn.BatchNorm1d(self.text_feat_dim)

        self.bn = nn.BatchNorm1d(3 * self.text_feat_dim)
        
        for head in [self.fc_r, self.fc_n, self.fc_t, self.fc, self.fc_r_t, self.fc_n_t, self.fc_t_t]:
            head.apply(weights_init_classifier)

        for bn in [self.bn_r, self.bn_n, self.bn_t, self.bn, self.bn_r_t, self.bn_n_t, self.bn_t_t]:
            bn.apply(weights_init_kaiming)

        self.dtype = clip_model.dtype
      
        print('Loading pretrained model from CLIP')
        
        self.base = clip_model
        
        model_path = '/data/gaoya/cot-reid/pretrained/ViT-B-16.pt'
        try:
        # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        state_dict = state_dict or model.state_dict()
        

        self.text_feat_dim = 512
    
        self.N0 = cfg.MODEL.CMI_N0
        self.N1 = cfg.MODEL.CMI_N1
        self.alpha = cfg.MODEL.CMI_ALPHA
        self.inference_fusion = cfg.MODEL.INFERENCE_FUSION
        self.inference_image_weight = cfg.MODEL.INFERENCE_IMAGE_WEIGHT
        self.inference_text_weight = cfg.MODEL.INFERENCE_TEXT_WEIGHT
        self.debug_text_encoding = cfg.MODEL.DEBUG_TEXT_ENCODING
        self._text_debug_printed = False
        self.align_loss_weight = cfg.MODEL.ALIGN_LOSS_WEIGHT
        self.align_temperature = cfg.MODEL.ALIGN_TEMPERATURE
        self.text_processor = PerModalTextProcessor(
            text_dim=512,       # 输入文本Token维度（r_cot_token为[B,768]）
            visual_dim=512,     # 输出维度
            num_modals=3        
        )

        self.cmi_prior = CMIPriorNetwork(context_dim=self.text_feat_dim)
        self.cmi_critics = nn.ModuleDict({
            f"pair_{pair_idx}_{direction}": CMICritic(
                feat_dim=self.text_feat_dim,
                context_dim=self.text_feat_dim,
            )
            for pair_idx in range(3)
            for direction in ("ab", "ba")
        })
        
        # 条件NCE判别器
        self.cmi_critic = nn.Sequential(
            nn.Conv2d(512*2, 256, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
        self.tokenizer = _Tokenizer()
        self.rgb_proj = nn.Linear(self.feat_dim, self.text_feat_dim)
        self.ni_proj = nn.Linear(self.feat_dim, self.text_feat_dim)
        self.ti_proj = nn.Linear(self.feat_dim, self.text_feat_dim)
        self.image_size = cfg.INPUT.SIZE_TRAIN

    def conditional_nce_with_text(self, visual_feats, modal_context_Ts, labels, N0=5, N1=1, prior_weight=0.1):
        """
        加入特征条件先验的条件CMI损失：用context_T定义先验分布，约束跨模态匹配概率
        prior_weight: 先验损失的权重
        """
        device = visual_feats[0].device
        input_dtype = visual_feats[0].dtype
        total_cmi_loss = 0.0
        total_prior_loss = 0.0  # 先验损失累计
        protective_margin = torch.log(torch.tensor(N1 / N0, device=device, dtype=input_dtype))

        if labels is None:
            raise ValueError("CMI training requires PID labels for valid negative sampling")
        labels = labels.reshape(-1).to(device)
        if labels.numel() != visual_feats[0].shape[0]:
            raise ValueError("CMI label count does not match the batch size")
        negative_mask = labels[:, None].ne(labels[None, :])
        if not negative_mask.any(dim=1).all():
            raise ValueError("Every CMI anchor needs at least one different-PID sample")
        negative_idx = torch.multinomial(
            negative_mask.to(torch.float32), N0, replacement=True
        )
        
        cross_modal_pairs = [(0,1), (1,2), (2,0)]
        
        for pair_idx, (idx_a, idx_b) in enumerate(cross_modal_pairs):
            feat_a = visual_feats[idx_a]  # [B, C]
            feat_b = visual_feats[idx_b]  # [B, C]
            T_a = modal_context_Ts[idx_a]  # [B, C_T]：条件T_a
            T_b = modal_context_Ts[idx_b]  # [B, C_T]：条件T_b
            B, C = feat_a.shape
            
            C_T = T_a.shape[1]
            critic_ab = self.cmi_critics[f"pair_{pair_idx}_ab"]
            critic_ba = self.cmi_critics[f"pair_{pair_idx}_ba"]
            
            # -------------------------- 正负样本构建 --------------------------
            pos_pair_ab = torch.cat([feat_a, feat_b, T_a], dim=1)  # [B, 2C+C_T]
            pos_pair_ba = torch.cat([feat_b, feat_a, T_b], dim=1)  # [B, 2C+C_T]
            
            # A→B负例
            neg_feat_a = feat_a[negative_idx].reshape(B*N0, C)
            pos_feat_b_repeat = feat_b.unsqueeze(1).repeat(1, N0, 1).reshape(B*N0, C)
            T_a_repeat = T_a.unsqueeze(1).repeat(1, N0, 1).reshape(B*N0, C_T)
            neg_pair_ab = torch.cat([neg_feat_a, pos_feat_b_repeat, T_a_repeat], dim=1)
            # B→A负例
            neg_feat_b = feat_b[negative_idx].reshape(B*N0, C)
            pos_feat_a_repeat = feat_a.unsqueeze(1).repeat(1, N0, 1).reshape(B*N0, C)
            T_b_repeat = T_b.unsqueeze(1).repeat(1, N0, 1).reshape(B*N0, C_T)
            neg_pair_ba = torch.cat([neg_feat_b, pos_feat_a_repeat, T_b_repeat], dim=1)
            
            # -------------------------- 计算匹配概率 --------------------------
            pos_prob_ab = critic_ab(pos_pair_ab).squeeze(-1)  # [B]：正样本概率
            neg_prob_ab = critic_ab(neg_pair_ab).reshape(B, N0)  # [B, N0]：负样本概率
            pos_prob_ba = critic_ba(pos_pair_ba).squeeze(-1)  # [B]
            neg_prob_ba = critic_ba(neg_pair_ba).reshape(B, N0)  # [B, N0]
            
            # -------------------------- 特征条件先验损失 --------------------------
            # 1. 用条件T计算先验分布参数
            (pos_mu_ab, pos_logvar_ab), (neg_mu_ab, neg_logvar_ab) = self.cmi_prior(T_a)
            (pos_mu_ba, pos_logvar_ba), (neg_mu_ba, neg_logvar_ba) = self.cmi_prior(T_b)
            
            # 2. 计算KL散度：预测分布 vs 先验分布（高斯假设下的KL）
            def kl_gaussian(pred, mu_prior, logvar_prior):
                """pred: 预测概率 [B, ...]；mu_prior, logvar_prior: 先验参数 [B]"""
                # 假设预测概率服从高斯分布 N(pred, 1e-3)（简化方差为固定小值）
                var_pred = torch.tensor(1e-3, device=device, dtype=input_dtype)
                logvar_pred = torch.log(var_pred)
                # KL(q||p) = 0.5*(log(var_prior/var_pred) + (var_pred + (pred - mu_prior)^2)/var_prior - 1)
                kl = 0.5 * (logvar_prior - logvar_pred + 
                           (var_pred + (pred - mu_prior)**2) / torch.exp(logvar_prior) - 1)
                return kl.mean()
            
            # A→B方向先验损失
            kl_pos_ab = kl_gaussian(pos_prob_ab, pos_mu_ab, pos_logvar_ab)  # 正样本匹配概率应贴近先验
            kl_neg_ab = kl_gaussian(neg_prob_ab.mean(dim=1), neg_mu_ab, neg_logvar_ab)  # 负样本均值贴近先验
            prior_loss_ab = kl_pos_ab + kl_neg_ab
            
            # B→A方向先验损失
            kl_pos_ba = kl_gaussian(pos_prob_ba, pos_mu_ba, pos_logvar_ba)
            kl_neg_ba = kl_gaussian(neg_prob_ba.mean(dim=1), neg_mu_ba, neg_logvar_ba)
            prior_loss_ba = kl_pos_ba + kl_neg_ba
            
            # 累计先验损失
            total_prior_loss += (prior_loss_ab + prior_loss_ba) / 2
            
            # -------------------------- 原NCE损失计算 --------------------------
            loss_ab = -torch.log(pos_prob_ab + 1e-8).mean() - (N0/N1)*torch.log(1 - neg_prob_ab + 1e-8).mean()
            loss_ba = -torch.log(pos_prob_ba + 1e-8).mean() - (N0/N1)*torch.log(1 - neg_prob_ba + 1e-8).mean()
            loss_bidirectional = (loss_ab + loss_ba) / 2 - protective_margin
            total_cmi_loss += loss_bidirectional
        
        # 总损失 = 原CMI损失 + 先验约束损失（加权）
        total_loss = (total_cmi_loss / len(cross_modal_pairs)) + \
                     (prior_weight * total_prior_loss / len(cross_modal_pairs))
        return total_loss, protective_margin
   
    
    def encode_text_global(self, token_ids):
        """Encode text and attention-pool every valid chunk.

        Padding chunks receive zero probability.  All non-padding chunks are
        retained and the shared scorer learns their relative importance.
        """
        if token_ids.dim() == 2:
            encoded = self.base.encode_text(token_ids)
            if token_ids.shape[1] > self.base.context_length:
                return encoded[:, 0]
            batch_idx = torch.arange(encoded.shape[0], device=encoded.device)
            return encoded[batch_idx, token_ids.argmax(dim=-1)]

        if token_ids.dim() != 3:
            raise ValueError(f"Expected text tokens with 2 or 3 dimensions, got {token_ids.shape}")

        batch_size, num_chunks, text_length = token_ids.shape
        flat_tokens = token_ids.reshape(batch_size * num_chunks, text_length)
        flat_encoded = self.base.encode_text(flat_tokens)
        flat_idx = torch.arange(flat_encoded.shape[0], device=flat_encoded.device)
        flat_eot = flat_encoded[flat_idx, flat_tokens.argmax(dim=-1)]
        chunk_features = flat_eot.reshape(batch_size, num_chunks, -1)

        valid_mask = token_ids.ne(0).any(dim=-1)
        attention_logits = self.chunk_attention(chunk_features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid_mask, float("-inf"))

        # tokenize_chunks always emits at least one valid chunk.  Keep this
        # fallback for malformed external batches to avoid softmax(NaN).
        no_valid_chunk = ~valid_mask.any(dim=1)
        if no_valid_chunk.any():
            attention_logits = attention_logits.clone()
            attention_logits[no_valid_chunk, 0] = 0.0

        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(-1)
        return (chunk_features * attention_weights).sum(dim=1)

    def _debug_text_encoding(self, name, token_ids, encoded_feature):
        """Print a compact integrity check for the first sample only."""
        sample = token_ids[0].detach().cpu()
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)

        sot = self.tokenizer.encoder["<|startoftext|>"]
        eot = self.tokenizer.encoder["<|endoftext|>"]
        valid_chunks = sample.ne(0).any(dim=1)
        print(f"[text-debug] {name}: shape={tuple(token_ids.shape)}, "
              f"valid_chunks={int(valid_chunks.sum())}, "
              f"feature_shape={tuple(encoded_feature.shape)}, "
              f"feature_norm={encoded_feature[0].float().norm().item():.4f}")

        for chunk_index in valid_chunks.nonzero(as_tuple=False).flatten().tolist():
            chunk = sample[chunk_index]
            non_padding = chunk[chunk.ne(0)].tolist()
            has_sot = bool(non_padding) and non_padding[0] == sot
            has_eot = bool(non_padding) and non_padding[-1] == eot
            content_tokens = non_padding[1:-1] if has_sot and has_eot else non_padding
            decoded = self.tokenizer.decode(content_tokens).strip()
            print(f"[text-debug]   chunk={chunk_index}, tokens={len(content_tokens)}, "
                  f"SOT={has_sot}, EOT={has_eot}, text={decoded!r}")


        def multi_positive_nce(scores, mask):
            positive_scores = scores.masked_fill(~mask, float("-inf"))
            return -(
                torch.logsumexp(positive_scores, dim=1)
                - torch.logsumexp(scores, dim=1)
            ).mean()

        image_to_text = multi_positive_nce(logits, positive_mask)
        text_to_image = multi_positive_nce(logits.t(), positive_mask.t())
        return 0.5 * (image_to_text + text_to_image)

    def forward(self, x, text, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None, layer=0):
        # -------------------------- 1. 多模态图像特征提取--------------------------------------------------
        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']
        RGB_Text = text['rgb_text']
        NI_Text = text['ni_text']
        TI_Text = text['ti_text']
        r_cot = text['r_cot']  # RGB模态专属文本Token（cot）
        n_cot = text['n_cot']  # NI模态专属文本Token（cot）
        t_cot = text['t_cot']  # TI模态专属文本Token（cot）
        
        # Descriptions are one sequence; CoT may contain several 77-token chunks.
        # Every chunk is represented by its EOT feature, then valid chunks are averaged.
        global_rgb_text = self.encode_text_global(RGB_Text)
        global_ni_text = self.encode_text_global(NI_Text)
        global_ti_text = self.encode_text_global(TI_Text)
        r_cot_token = self.encode_text_global(r_cot)
        n_cot_token = self.encode_text_global(n_cot)
        t_cot_token = self.encode_text_global(t_cot)

        if self.debug_text_encoding and not self._text_debug_printed:
            debug_items = [
                ("rgb_text", RGB_Text, global_rgb_text),
                ("ni_text", NI_Text, global_ni_text),
                ("ti_text", TI_Text, global_ti_text),
                ("r_cot", r_cot, r_cot_token),
                ("n_cot", n_cot, n_cot_token),
                ("t_cot", t_cot, t_cot_token),
            ]
            for name, tokens, feature in debug_items:
                self._debug_text_encoding(name, tokens, feature)
            self._text_debug_printed = True
        
        # 文本Token投影到视觉特征维度
        R_proj = self.text_token_proj(r_cot_token)  # [B,768]→[B,512]
        N_proj = self.text_token_proj(n_cot_token)  # [B,768]→[B,512]
        T_proj = self.text_token_proj(t_cot_token)  # [B,768]→[B,512]

      
        # 视觉模态特征提取
        RGB_cash, RGB_global = self.RGB_BACKBONE(RGB, R_proj, cam_label=cam_label, view_label=view_label)
        NI_cash, NI_global = self.NI_BACKBONE(NI, N_proj, cam_label=cam_label, view_label=view_label)
        TI_cash, TI_global = self.TI_BACKBONE(TI, T_proj, cam_label=cam_label, view_label=view_label)

        # 视觉特征投影到文本维度（
        RGB_proj = self.rgb_proj(RGB_global)  # [B,768]→[B,512]
        NI_proj = self.ni_proj(NI_global)     # [B,768]→[B,512]
        TI_proj = self.ti_proj(TI_global)     # [B,768]→[B,512]
    
        per_modal_cot_tokens = [r_cot_token, n_cot_token, t_cot_token]  # 列表长度3，每个元素[B,768]
       # pdb.set_trace()
        per_modal_text_modals, per_modal_context_T = self.text_processor(per_modal_cot_tokens)
      
        R_text_modal = per_modal_text_modals[0]  # RGB专属模态化文本特征 [B,512,1,1]
        N_text_modal = per_modal_text_modals[1]  # NI专属模态化文本特征 [B,512,1,1]
        T_text_modal = per_modal_text_modals[2]  # TI专属模态化文本特征 [B,512,1,1]
        r_text_T = per_modal_context_T[0]        # RGB专属上下文T [B,512]
        n_text_T = per_modal_context_T[1]        # NI专属上下文T [B,512]
        t_text_T = per_modal_context_T[2]        # TI专属上下文T [B,512]
        modal_context_Ts = [r_text_T, n_text_T, t_text_T]

        modal_cross_pair = [RGB_proj, NI_proj, TI_proj]
        
        # -------------------------- 3.CMI优化输入--------------------------
        # 每个视觉模态投影特征 ↔ 其专属模态化文本特征（维度统一为[B,512,1,1]）
        # -------------------------- 4. 多模态特征拼接（ --------------------------
        cls = torch.cat([RGB_proj, NI_proj, TI_proj], dim=-1)
        
        # -------------------------- 5. 训练阶段：损失计算 --------------------------
        if self.training:
            # reid_loss
            rgb_id = self.fc_r(self.bn_r(RGB_proj))
            ni_id = self.fc_n(self.bn_n(NI_proj))
            ti_id = self.fc_t(self.bn_t(TI_proj))
            ori_id = self.fc(self.bn(cls))
            # cot_text_loss
            rgb_text_id = self.fc_r_t(self.bn_r_t(global_rgb_text))
            ni_text_id  = self.fc_n_t(self.bn_n_t(global_ni_text))
            ti_text_id  = self.fc_t_t(self.bn_t_t(global_ti_text))
            #cmi_loss
            cmi_Loss, _ = self.conditional_nce_with_text(
                visual_feats=modal_cross_pair,
                modal_context_Ts=modal_context_Ts,
                labels=label,
                N0=self.N0,
                N1=self.N1
            )

            
            total_loss = self.alpha * cmi_Loss
            
        # pdb.set_trace()
            return [rgb_id, ni_id, ti_id, rgb_text_id, ni_text_id, ti_text_id, ori_id], [RGB_proj, NI_proj, TI_proj, global_rgb_text, global_ni_text, global_ti_text, cls], total_loss
        else:
            if self.inference_fusion == 'legacy':
                ori_f = torch.cat([
                    RGB_proj, NI_proj, TI_proj,
                    global_rgb_text, global_ni_text, global_ti_text, cls
                ], dim=-1)
            elif self.inference_fusion == 'balanced':
                image_features = [RGB_proj, NI_proj, TI_proj]
                text_features = [global_rgb_text, global_ni_text, global_ti_text]
                image_features = [
                    F.normalize(feature.float(), p=2, dim=1)
                    * self.inference_image_weight
                    for feature in image_features
                ]
                text_features = [
                    F.normalize(feature.float(), p=2, dim=1)
                    * self.inference_text_weight
                    for feature in text_features
                ]
                ori_f = torch.cat(image_features + text_features, dim=-1)
            else:
                raise ValueError(
                    f"Unsupported inference fusion: {self.inference_fusion!r}. "
                    "Expected 'legacy' or 'balanced'."
                )
        
            return ori_f
       
    
    def load_param(self, model_path, map_location=None):
        checkpoint = torch.load(
            model_path,
            map_location=map_location if map_location is not None else torch.device('cpu'),
        )
        if 'model_state_dict' in checkpoint:
            # 格式1：有外层键'model_state_dict'
            param_dict = checkpoint['model_state_dict']
        else:
            # 格式2：直接是模型参数（无外层键）
            param_dict = checkpoint  # 直接使用整个checkpoint作为参数字典
        
        # 处理多GPU训练的 'module.' 前缀（如果有）
        param_dict = {k.replace('module.', ''): v for k, v in param_dict.items()}
        
        # 只加载模型中存在且形状一致的参数。
        model_dict = self.state_dict()
        valid_params = {
            k: v for k, v in param_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # 打印未匹配的参数（用于调试）
        missing_keys = [k for k in param_dict if k not in model_dict]
        if missing_keys:
            print(f"警告：以下参数未在模型中找到，已跳过：{missing_keys}")

        mismatched_keys = [
            (k, tuple(v.shape), tuple(model_dict[k].shape))
            for k, v in param_dict.items()
            if k in model_dict and v.shape != model_dict[k].shape
        ]
        if mismatched_keys:
            print(f"警告：以下参数形状不匹配，已跳过：{mismatched_keys}")
        
        # 加载有效参数
        model_dict.update(valid_params)
        self.load_state_dict(model_dict)
        print("模型权重加载成功！")
__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}
class EarlyFusionLayer(nn.Module):
    def __init__(self, img_feat_dim=1024, prompt_dim=512, num_heads=8):
        super().__init__()
        """
        Args:
    
        """
        # 1. 特征维度统一（图像特征→Prompt维度）
        self.img_proj = nn.Linear(img_feat_dim, prompt_dim)
        self.img_proj.apply(weights_init_kaiming)  # 论文中权重初始化方式

        self.gate_attn_img2text = nn.MultiheadAttention(
            embed_dim=prompt_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate_attn_text2img = nn.MultiheadAttention(
            embed_dim=prompt_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.background_token = nn.Parameter(torch.randn(1, 1, prompt_dim))
        nn.init.normal_(self.background_token, std=0.02)

        self.norm1 = nn.LayerNorm(prompt_dim)
        self.norm2 = nn.LayerNorm(prompt_dim)
        self.ffn = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim * 2),
            nn.ReLU(),
            nn.Linear(prompt_dim * 2, prompt_dim)
        )

    def gated_attention(self, Q, K, V):
        # 拼接K/V与背景token [B, seq_len+1, dim]
        K_with_bg = torch.cat([K, self.background_token.expand(K.shape[0], -1, -1)], dim=1)
        V_with_bg = torch.cat([V, self.background_token.expand(V.shape[0], -1, -1)], dim=1)
        
        # 注意力计算
        attn_output, _ = self.gate_attn_img2text(
            query=Q,
            key=K_with_bg,
            value=V_with_bg
        )
        return attn_output

    def forward(self, img_feat, prompt_feat):
        """
        Args:
            img_feat: 单模态图像特征（如RGB，[B, img_feat_dim]）
            prompt_feat: Prompt特征（文本/视觉，[B, prompt_dim]）
        Returns:
            fused_feat: 融合后特征 [B, prompt_dim]
        """
        # 1. 图像特征维度统一
        img_feat_proj = self.img_proj(img_feat).unsqueeze(1)  # [B, 1, prompt_dim]
        prompt_feat_expand = prompt_feat.unsqueeze(1)  # [B, 1, prompt_dim]

        # 2. 双向门控注意力
        prompt_updated = self.gated_attention(
            Q=prompt_feat_expand,
            K=img_feat_proj,
            V=img_feat_proj
        )
        prompt_updated = self.norm1(prompt_feat_expand + prompt_updated)

        img_updated = self.gated_attention(
            Q=img_feat_proj,
            K=prompt_updated,
            V=prompt_updated
        )
        img_updated = self.norm2(img_feat_proj + img_updated)

        # 3. FFN细化融合特征
        fused_feat = self.ffn(img_updated.squeeze(1))  # [B, prompt_dim]
        fused_feat = F.normalize(fused_feat, dim=1)

        return fused_feat
    


def make_model(cfg, num_class, camera_num, view_num=0):
    model = CoTReID(num_class, cfg, camera_num, view_num, __factory_T_type)
    # total_flops = model.flops()  # 默认shape，自动适配模型image_size
    # print(f"\n模型总FLOPs：{total_flops:.2f} FLOPs")
    # print(f"模型总GFLOPs：{total_flops / 1e9:.2f} GFLOPs")
    print('===========Building CoT-ReID===========')
    return model


def compute_cross_modal_entropy(modal_features, temperature=0.07, eps=1e-8):
    """Compute mean bidirectional matching entropy across modality pairs."""
    if len(modal_features) < 2:
        raise ValueError("At least two modal features are required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    normalized_features = []
    batch_size = modal_features[0].shape[0]
    feature_dim = modal_features[0].reshape(batch_size, -1).shape[1]
    for feature in modal_features:
        flattened = feature.reshape(feature.shape[0], -1).float()
        if flattened.shape != (batch_size, feature_dim):
            raise ValueError("All modal features must have matching batch and feature dimensions")
        normalized_features.append(F.normalize(flattened, p=2, dim=1))

    pair_entropies = []
    for source_index in range(len(normalized_features)):
        for target_index in range(source_index + 1, len(normalized_features)):
            logits = (
                normalized_features[source_index]
                @ normalized_features[target_index].t()
            ) / temperature
            source_prob = F.softmax(logits, dim=1)
            target_prob = F.softmax(logits.t(), dim=1)
            source_entropy = -(source_prob * source_prob.clamp_min(eps).log()).sum(dim=1).mean()
            target_entropy = -(target_prob * target_prob.clamp_min(eps).log()).sum(dim=1).mean()
            pair_entropies.append(0.5 * (source_entropy + target_entropy))

    return torch.stack(pair_entropies).mean()

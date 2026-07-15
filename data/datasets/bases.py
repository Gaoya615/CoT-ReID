
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
# The text encoder is CLIP, so dataset token IDs must come from the exact same
# CLIP vocabulary.  The former utils.simple_tokenizer has a separate 960-token
# vocabulary and produced IDs that CLIP interpreted as unrelated BPE tokens.
from modeling.clip.simple_tokenizer import SimpleTokenizer


class BaseDataset(object):
    """
    Base class of reid dataset
    """
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for item in data:
            # 支持两种格式：
            # 1) (img_path, pid, camid)  -> 常规单图
            # 2) ((rgb_path, ni_path, ti_path), pid, camid, trackid, ... ) -> 多模态/tracklet 风格
            try:
                if isinstance(item[0], (list, tuple)) and len(item) >= 4:
                    pid = item[1]
                    camid = item[2]
                    trackid = item[3]
                else:
                    # 兼容传统三元组
                    _, pid, camid = item
                    trackid = None
            except Exception:
                # 如果格式不符合预期，尽量解析常见位置
                if len(item) >= 3:
                    pid = item[1]
                    camid = item[2]
                else:
                    continue
                trackid = None

            pids.append(pid)
            cams.append(camid)
            tracks.append(trackid)

        pids = set(pids)
        cams = set(cams)
        # track id 可能为 None，统计非 None 的不同 track
        tracks_set = set([t for t in tracks if t is not None])
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_tracks = len(tracks_set) if len(tracks_set) > 0 else 0
        return num_pids, num_imgs, num_cams, num_tracks

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """
    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, _ = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


# =========================================================================
# 核心数据加载器：ImageDataset
# =========================================================================

def read_image(img_path):
    """读取单张图片，带有容错机制"""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo.".format(img_path))
            pass
    return img

def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
    result = torch.zeros(text_length, dtype=torch.long)
    
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(f"Input too long for length {text_length}")
            
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageDataset(Dataset):
    """
    为多模态 (RGB, NI, TI) + 属性 + 文本 + 综合思维链 设计的数据集包装器。
    使用 SimpleTokenizer 直接输出对齐的 Tensor。
    """
    def __init__(self, dataset, transform=None, text_length: int = 77, truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 1. 拿到当前索引的数据元组
        data_tuple = self.dataset[index]

        if len(data_tuple) == 10:
            img_paths, pid, camid, sceneid, \
            text_rgb, text_ni, text_ti, \
            rgb_cot_full, ni_cot_full, ti_cot_full = data_tuple
            
        else:
            raise ValueError(f"致命错误：数据集返回了 {len(data_tuple)} 个元素，期待 10 个")

        rgb_path, ni_path, ti_path = img_paths

        # 3. 读取并转换图像
        img_rgb = read_image(rgb_path)
        img_ni = read_image(ni_path)
        img_ti = read_image(ti_path)

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_ni = self.transform(img_ni)
            img_ti = self.transform(img_ti)

        imgs = (img_rgb, img_ni, img_ti)

        # 4.  使用自定义 tokenize 函数转换所有文本，直接吐出 Tensor
        r_tokens = tokenize(text_rgb, self.tokenizer, self.text_length, self.truncate)
        n_tokens = tokenize(text_ni, self.tokenizer, self.text_length, self.truncate)
        t_tokens = tokenize(text_ti, self.tokenizer, self.text_length, self.truncate)

        r_cot_tokens = tokenize(rgb_cot_full, self.tokenizer, self.text_length, self.truncate)
        n_cot_tokens = tokenize(ni_cot_full, self.tokenizer, self.text_length, self.truncate)
        t_cot_tokens = tokenize(ti_cot_full, self.tokenizer, self.text_length, self.truncate)

        # 5. 返回数据 (共 10 个元素)
        return (
            imgs, pid, camid, sceneid, index, img_paths,
            r_tokens, n_tokens, t_tokens,
            r_cot_tokens, n_cot_tokens, t_cot_tokens
        )


def tokenize_chunks(caption, tokenizer, text_length=77, max_chunks=4):
    """Keep the complete CLIP token sequence for model-side windowing."""
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot] + tokenizer.encode(caption) + [eot]
    result = torch.zeros(max(text_length, len(tokens)), dtype=torch.long)
    result[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result


class LongTextCoTImageDataset(ImageDataset):
    """Image dataset retaining full descriptions for CLIP-side windows."""

    def __init__(
        self,
        dataset,
        transform=None,
        text_length=77,
        truncate=True,
        max_description_chunks=8,
        max_cot_chunks=9,
    ):
        super().__init__(dataset, transform, text_length, truncate)
        self.max_description_chunks = max_description_chunks
        self.max_cot_chunks = max_cot_chunks

    def __getitem__(self, index):
        sample = list(super().__getitem__(index))
        data_tuple = self.dataset[index]
        rgb_text, ni_text, ti_text = data_tuple[4:7]
        rgb_cot, ni_cot, ti_cot = data_tuple[7:10]
        sample[-6] = tokenize_chunks(rgb_text, self.tokenizer, self.text_length, self.max_description_chunks)
        sample[-5] = tokenize_chunks(ni_text, self.tokenizer, self.text_length, self.max_description_chunks)
        sample[-4] = tokenize_chunks(ti_text, self.tokenizer, self.text_length, self.max_description_chunks)
        sample[-3] = tokenize_chunks(rgb_cot, self.tokenizer, self.text_length, self.max_cot_chunks)
        sample[-2] = tokenize_chunks(ni_cot, self.tokenizer, self.text_length, self.max_cot_chunks)
        sample[-1] = tokenize_chunks(ti_cot, self.tokenizer, self.text_length, self.max_cot_chunks)
        return tuple(sample)

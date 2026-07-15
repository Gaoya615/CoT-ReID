# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from config import cfg
import os
import os.path as osp
from .bases import BaseImageDataset
import json

import pdb

class MSVR310_text(BaseImageDataset):

    dataset_dir = 'MSVR310'
    train_text_suffix = ''
    test_text_suffix = ''

    def __init__(self, root='', verbose=True,  cfg=cfg, **kwargs):
        super(MSVR310_text, self).__init__()
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.prompt = cfg.MODEL.TEXT_PROMPT * 'X ' if cfg.MODEL.TEXT_PROMPT > 0 else ''
        self.text_dir = cfg.DATASETS.TEXT_DIR or getattr(
            self, "text_dir_override", "/data/gaoya/cot-reid/text/MSVR310"
        )
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query3')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.train_text_dir = osp.join(self.text_dir, '')
        self.query_text_dir =  osp.join(self.text_dir, '')
        self.gallery_text_dir = osp.join(self.text_dir, '')

        self._check_before_run()

        train = self._process_dir(self.train_dir, self.train_text_dir, relabel=True)
        query = self._process_dir(self.query_dir, self.query_text_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_text_dir, relabel=False)

        if verbose:
            print("=> RGB_IR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def find_annotation(self, annotation_list, image_name):
        """从JSON数组中直接提取 cot_description 文本作为综合描述"""
        # 校验输入格式
        if not isinstance(annotation_list, list):
            print(f"标注格式错误：预期数组，实际是 {type(annotation_list).__name__}（图像：{image_name}）")
            return ""
        for vehicle_obj in annotation_list:
            if not isinstance(vehicle_obj, dict):
                continue
            if vehicle_obj.get("filename") == image_name:
                # 提取 cot_description 并容错处理
                cot_desc = vehicle_obj.get("cot_description", "")
                if isinstance(cot_desc, str) and cot_desc.strip():
                    return cot_desc.strip()
                else:
                    print(f"警告：{image_name} 的 cot_description 为空或字段不存在")
                    return ""
        
        # 未匹配提示
        print(f"未找到{image_name}的标注（数组共{len(annotation_list)}个对象）")
        return ""

    def find_annotation_cot(self, annotation_list, image_name):
        """从JSON数组中提取 reasoning_chain"""
        if not isinstance(annotation_list, list):
            print(f" COT标注格式错误：预期数组，实际是 {type(annotation_list).__name__}（图像：{image_name}）")
            return ""

        # 按filename精准匹配
        for vehicle_obj in annotation_list:
            if not isinstance(vehicle_obj, dict):
                continue
            if vehicle_obj.get("filename") == image_name:
                annot_parts = []

                # 提取reasoning_chain（增加类型校验）
                reasoning_chain = vehicle_obj.get("reasoning_chain")
                if isinstance(reasoning_chain, str):
                    reasoning_text = reasoning_chain.strip()
                    annot_parts.append(f"Reasoning: {reasoning_text}")
                else:
                    print(f"{image_name}的reasoning_chain不是字符串（类型：{type(reasoning_chain).__name__}）")

                return " ; ".join(annot_parts).strip()
        
        print(f"未找到{image_name}的COT标注（数组共{len(annotation_list)}个对象）")
        return ""
   
    def _process_dir(self, dir_path, text_dir_path, relabel=False):
        prefix = 'train' if 'train' in dir_path else 'test'
        suffix = self.train_text_suffix if prefix == 'train' else self.test_text_suffix

        json_file_RGB = osp.join(text_dir_path, f'{prefix}_RGB{suffix}.json')
        json_file_NI = osp.join(text_dir_path, f'{prefix}_NI{suffix}.json')
        json_file_TI = osp.join(text_dir_path, f'{prefix}_TI{suffix}.json')
          
        with open(json_file_RGB, 'r', encoding='utf-8') as f_rgb:
            text_annotations_RGB = json.load(f_rgb)
            if not isinstance(text_annotations_RGB, list):
                raise TypeError(f"{json_file_RGB}解析后应为list，实际是{type(text_annotations_RGB).__name__}")
        with open(json_file_NI, 'r', encoding='utf-8') as f_ni:
            text_annotations_NI = json.load(f_ni)
            if not isinstance(text_annotations_NI, list):
                raise TypeError(f"{json_file_NI}解析后应为list，实际是{type(text_annotations_NI).__name__}")
        with open(json_file_TI, 'r', encoding='utf-8') as f_ti:
            text_annotations_TI = json.load(f_ti)
            if not isinstance(text_annotations_TI, list):
                raise TypeError(f"{json_file_TI}解析后应为list，实际是{type(text_annotations_TI).__name__}")
        
        vid_container = set()
        for vid in os.listdir(dir_path):
            vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)}
    
        dataset = []
        for vid in os.listdir(dir_path):
            vid_path = osp.join(dir_path, vid)
            r_data = os.listdir(osp.join(vid_path, 'vis'))
            for img in r_data:
                r_img_path = osp.join(vid_path, 'vis', img)
                n_img_path = osp.join(vid_path, 'ni', img)
                t_img_path = osp.join(vid_path, 'th', img)
    
                try:
                    # 提取车辆ID（从图片名前4位）
                    vehicle_id = img[:4]
                    vid = int(vehicle_id)
                    camid = int(img[11]) if len(img) > 11 else 0
                    sceneid = int(img[6:9]) if len(img) > 9 else 0
                except (IndexError, ValueError):
                    print(f" 图像名格式异常：{img}，跳过处理")
                    continue
    
                if relabel:
                    vid = vid2label[vid]
                jpg_name = img
                match_name = img  

                # 调用标注查找函数（由于 find_annotation 被修改，这里获取的直接是 cot_description）
                rgb_annot = self.find_annotation(text_annotations_RGB, match_name)
                ni_annot = self.find_annotation(text_annotations_NI, match_name)
                ti_annot = self.find_annotation(text_annotations_TI, match_name)
    
                rgb_annot_cot = 'An image of a vehicle in the visible spectrum, capturing natural colors and fine details: '+self.find_annotation_cot(text_annotations_RGB, match_name)
                ni_annot_cot = 'An image of a vehicle in the near infrared spectrum, capturing contrasts and surface reflectance: '+self.find_annotation_cot(text_annotations_NI, match_name)
                ti_annot_cot = 'An image of a vehicle in the thermal infrared spectrum, capturing heat emissions as temperature gradients: '+ self.find_annotation_cot(text_annotations_TI, match_name)
    
                # 拼接文本标注（前缀保持不变，后接提取出的 cot_description）
                text_annotation_RGB = rgb_annot
                text_annotation_NI = ni_annot
                text_annotation_TI = ti_annot

                # print(f"rgb_cot:{rgb_annot_cot}")
                # print(f"rgb_text:{rgb_annot}")
                dataset.append(
                    (
                        (r_img_path, n_img_path, t_img_path),
                        vid, camid, sceneid,
                        text_annotation_RGB,
                        text_annotation_NI,
                        text_annotation_TI,
                        rgb_annot_cot,
                        ni_annot_cot,
                        ti_annot_cot,
                    )
                )
        return dataset

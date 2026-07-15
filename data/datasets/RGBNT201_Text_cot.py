from __future__ import division, print_function, absolute_import
import glob
import warnings
import os.path as osp
from .bases import BaseImageDataset
import json
from config import cfg


class RGBNT201_Text(BaseImageDataset):
    dataset_dir = 'RGBNT201'

    def __init__(self, root='', verbose=True, cfg=cfg, **kwargs):
        super(RGBNT201_Text, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.prompt = cfg.MODEL.TEXT_PROMPT * 'X ' if cfg.MODEL.TEXT_PROMPT > 0 else ''
        # self.prefix = cfg.MODEL.PREFIX
        # if self.prefix:
        #     print('~~~~~~~【We use modality prefix Here!】~~~~~~~')
        # else:
        #     print('~~~~~~~【We do not use modality prefix Here!】~~~~~~~')
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.text_dir = cfg.DATASETS.TEXT_DIR or "/data/gaoya/cot-reid/text/RGBNT201"
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated.'
            )

        self.train_dir = osp.join(self.data_dir, 'train_171')
        self.query_dir = osp.join(self.data_dir, 'test')
        self.gallery_dir = osp.join(self.data_dir, 'test')

        self.train_text_dir = osp.join(self.text_dir, '')
        self.query_text_dir = osp.join(self.text_dir, '')
        self.gallery_text_dir = osp.join(self.text_dir, '')

        self._check_before_run()

        train = self._process_dir(self.train_dir, self.train_text_dir, relabel=True)
        query = self._process_dir(self.query_dir, self.query_text_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_text_dir, relabel=False)
        if verbose:
            print("=> RGBNT201 loaded")
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

    def _get_annotation_item(self, annotation_dict, image_name):
        if not isinstance(annotation_dict, dict):
            print(f" 标注格式错误：预期dict，实际是{type(annotation_dict).__name__}（图像：{image_name}）")
            return None

        pure_image_name = osp.basename(image_name)
        annot = annotation_dict.get(pure_image_name)
        if annot is None:
            print(f" 未找到{pure_image_name}的标注（标注字典共{len(annotation_dict)}条记录）")
            return None
        return annot

    def find_annotation(self, annotation_dict, image_name):
        annot = self._get_annotation_item(annotation_dict, image_name)
        if annot is None:
            return ""

        comprehensive_description = annot.get("comprehensive_description", "")
        if isinstance(comprehensive_description, str) and comprehensive_description.strip():
            return comprehensive_description.strip()

        features = annot.get("features", [])
        if isinstance(features, list):
            features_text = "; ".join([
                f"{f['name']}: {f['description']}"
                for f in features
                if isinstance(f, dict) and "name" in f and "description" in f
            ])
            return features_text.strip()

        return ""
    
    def find_annotation_cot(self, annotation_dict, image_name):
        annot = self._get_annotation_item(annotation_dict, image_name)
        if annot is None:
            return ""

        reasoning_chain = annot.get("reasoning_chain", "")
        if isinstance(reasoning_chain, str):
            return reasoning_chain.strip()
        return ""

    def _process_dir(self, dir_path, text_dir_path, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        if len(img_paths_RGB) == 0:
            raise RuntimeError(f"'{dir_path}/RGB' 下无jpg图片（图片路径错误）")

        text_annotations_RGB = {}
        text_annotations_NI = {}
        text_annotations_TI = {}
        
        prefix = 'train' if 'train' in dir_path else 'test'
        suffix = ""
        json_file_RGB = osp.join(text_dir_path, f"{prefix}_RGB{suffix}.json")
        json_file_NI = osp.join(text_dir_path, f"{prefix}_NI{suffix}.json")
        json_file_TI = osp.join(text_dir_path, f"{prefix}_TI{suffix}.json")
            
        with open(json_file_RGB, 'r', encoding='utf-8') as f_rgb:
            text_annotations_RGB = json.load(f_rgb)
        with open(json_file_NI, 'r', encoding='utf-8') as f_ni:
            text_annotations_NI = json.load(f_ni)
        with open(json_file_TI, 'r', encoding='utf-8') as f_ti:
            text_annotations_TI = json.load(f_ti)

        pid_container = set()
        for img_path in img_paths_RGB:
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[0][:6])
            pid_container.add(pid)
        pid2label = {pid: idx for idx, pid in enumerate(pid_container)} if relabel else None
        
        data = []
        for img_path_RGB in img_paths_RGB:
            img_name = osp.basename(img_path_RGB)
            img_path_NI = osp.join(dir_path, 'NI', img_name)
            img_path_TI = osp.join(dir_path, 'TI', img_name)
            if not osp.exists(img_path_NI):
                warnings.warn(f"NI图片 {img_path_NI} 不存在，跳过该样本")
                continue
            if not osp.exists(img_path_TI):
                warnings.warn(f"TI图片 {img_path_TI} 不存在，跳过该样本")
                continue
        
            pid = int(img_name.split('_')[0][:6])
            camid = int(img_name.split('_')[1][3]) - 1
            trackid = -1
        
            if relabel:
                pid = pid2label[pid]

            rgb_text = self.find_annotation(text_annotations_RGB, img_path_RGB)
            ni_text = self.find_annotation(text_annotations_NI, img_path_NI)
            ti_text = self.find_annotation(text_annotations_TI, img_path_TI)
    
            rgb_annot_cot = (
                'An image of a person in the visible spectrum, capturing natural colors and fine details: '
                + self.find_annotation_cot(text_annotations_RGB, img_path_RGB)
            )
            ni_annot_cot = (
                'An image of a person in the near infrared spectrum, capturing contrasts and surface reflectance: '
                + self.find_annotation_cot(text_annotations_NI, img_path_NI)
            )
            ti_annot_cot = (
                'An image of a person in the thermal infrared spectrum, capturing heat emissions as temperature gradients: '
                + self.find_annotation_cot(text_annotations_TI, img_path_TI)
            )
            
            text_annotation_RGB = rgb_text
            text_annotation_NI = ni_text
            text_annotation_TI = ti_text

            data.append(((img_path_RGB, img_path_NI, img_path_TI), pid, camid, trackid, text_annotation_RGB,
                                text_annotation_NI, text_annotation_TI, rgb_annot_cot, ni_annot_cot, ti_annot_cot))
                
        return data

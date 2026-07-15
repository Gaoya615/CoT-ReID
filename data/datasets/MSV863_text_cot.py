from __future__ import division, print_function, absolute_import
import os
import os.path as osp
import re
import warnings
import json
from config import cfg
from .bases import BaseImageDataset

class MSV863_Text(BaseImageDataset):
    dataset_dir = 'WMVEID863'  # 与MSVR310_Text保持目录命名逻辑一致
    train_text_suffix = ""
    test_text_suffix = ""

    def __init__(self, root='', verbose=True, cfg=cfg, **kwargs):
        super(MSV863_Text, self).__init__()
        # 统一根目录处理
        root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.prompt = cfg.MODEL.TEXT_PROMPT * 'X ' if cfg.MODEL.TEXT_PROMPT > 0 else ''
        
        # 文本标注目录
        self.text_dir = getattr(
            self,
            "text_dir_override",
            "/data/gaoya/cot-reid/text/MSV863",
        )
        self.train_text_dir = self.text_dir
        self.query_text_dir = self.text_dir
        self.gallery_text_dir = self.text_dir
        
        # 数据集子目录
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.query_dir = osp.join(self.dataset_dir, 'query')

        # 目录存在性检查
        self._check_before_run()

        # 数据处理入口
        train = self._process_dir(self.train_dir, self.train_text_dir, relabel=True)
        query = self._process_dir(self.query_dir, self.query_text_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_text_dir, relabel=False)

        # 日志打印
        if verbose:
            print("=> WMVEID863 loaded")
            self.print_dataset_statistics(train, query, gallery)

        # 数据集属性赋值
        self.train = train
        self.query = query
        self.gallery = gallery

        # 统计信息计算
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """检查核心目录是否存在"""
        required_dirs = [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir, self.text_dir]
        for dir_path in required_dirs:
            if not osp.exists(dir_path):
                raise RuntimeError(f"'{dir_path}' is not available (目录不存在)")

    def _load_json_annotations(self, file_path):
        """加载JSON标注文件，强制校验数组格式"""
        if not osp.exists(file_path):
            raise RuntimeError(f"标注文件不存在：{file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON格式错误：{file_path} 应为顶层数组，实际是 {type(data).__name__}")
        return data

    def _extract_vehicle_id_from_name(self, image_name):
        match = re.match(r'^(\d{4})_', image_name)
        return match.group(1) if match else None

    def _build_annotation_store(self, annotation_list):
        """构建标注索引，支持精确匹配、别名匹配和按车辆回退匹配"""
        exact = {}
        by_vehicle = {}
        for item in annotation_list:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                continue
            exact[filename] = item
            vehicle_id = self._extract_vehicle_id_from_name(filename) or str(item.get("vehicle_id", "")).zfill(4)
            by_vehicle.setdefault(vehicle_id, []).append(item)

        for vehicle_id in by_vehicle:
            by_vehicle[vehicle_id].sort(key=lambda x: x.get("filename", ""))

        return {"exact": exact, "by_vehicle": by_vehicle}

    def _candidate_annotation_names(self, image_name):
        """为异常命名样本生成一组候选文件名"""
        candidates = [image_name]
        replace_pairs = [
            ("_vis_", "_ni_"),
            ("_vis_", "_th_"),
            ("_ni_", "_vis_"),
            ("_ni_", "_th_"),
            ("_th_", "_vis_"),
            ("_th_", "_ni_"),
        ]
        for src, dst in replace_pairs:
            if src in image_name:
                candidates.append(image_name.replace(src, dst))
        return candidates

    def _resolve_annotation_item(self, annotation_store, image_name, vehicle_id=None, pair_index=None):
        """优先精确匹配，其次尝试命名别名，最后按同车辆排序位置回退"""
        exact = annotation_store["exact"]
        for candidate in self._candidate_annotation_names(image_name):
            item = exact.get(candidate)
            if item is not None:
                return item

        if vehicle_id is not None:
            vehicle_items = annotation_store["by_vehicle"].get(str(vehicle_id).zfill(4), [])
            if pair_index is not None and 0 <= pair_index < len(vehicle_items):
                return vehicle_items[pair_index]

        return None

    def find_annotation(self, annotation_list, image_name):
        """从JSON数组中提取 features + feature_relations 拼接文本（增加容错处理）"""
        # 校验输入格式
        if not isinstance(annotation_list, list):
            print(f"标注格式错误：预期数组，实际是 {type(annotation_list).__name__}（图像：{image_name}）")
            return ""
        
        # 按filename精准匹配
        for vehicle_obj in annotation_list:
            if not isinstance(vehicle_obj, dict):
                continue
            if vehicle_obj.get("filename") == image_name:
                print(f" 找到{image_name}的标注（数组共{len(annotation_list)}个对象）")
                annot_parts = []

                # 提取features（保留原有格式，增加空列表容错）
                if "features" in vehicle_obj and isinstance(vehicle_obj["features"], list):
                    if len(vehicle_obj["features"]) == 0:
                        print(f" {image_name}的features为空列表")
                    else:
                        features_text = "; ".join([
                            f"{f['name']}: {f['description']}" 
                            for f in vehicle_obj["features"] 
                            if isinstance(f, dict) and "name" in f and "description" in f
                        ])
                        annot_parts.append(features_text)

                # 提取feature_relations（核心修复：增加类型校验和键存在性检查）
                if "feature_relations" in vehicle_obj and isinstance(vehicle_obj["feature_relations"], list):
                    relations_list = []
                    for idx, r in enumerate(vehicle_obj["feature_relations"]):
                        # 跳过非字典元素
                        if not isinstance(r, dict):
                            print(f"{image_name}的feature_relations第{idx+1}个元素不是字典（类型：{type(r).__name__}），跳过")
                            continue
                        # 检查必要键是否存在，缺失则用默认值
                        relation_name = r.get("relation", "Unnamed relation")
                        relation_desc = r.get("description", "No description")
                        relation_value = r.get("reid_value", "No ReID value")
                        # 拼接单个关系文本
                        relations_list.append(
                            f"Relation: {relation_name} - {relation_desc} (ReID value: {relation_value})"
                        )
                    # 若有有效关系，添加到标注中
                    if relations_list:
                        relations_text = "; ".join(relations_list)
                        annot_parts.append(relations_text)
                    else:
                        print(f" {image_name}的feature_relations无有效字典元素")

                # 拼接所有部分（去重空字符串）
                final_text = "; ".join([part for part in annot_parts if part.strip()])
                return final_text.strip()
        
        # 未匹配提示（打印完整图像名，方便调试）
        print(f" 未找到{image_name}的标注（数组共{len(annotation_list)}个对象）")
        return ""

    def find_annotation_cot(self, annotation_list, image_name):
        """从JSON数组中提取 reasoning_chain（适配数组格式，增加容错）"""
        if not isinstance(annotation_list, list):
            print(f"COT标注格式错误：预期数组，实际是 {type(annotation_list).__name__}（图像：{image_name}）")
            return ""

        # 按filename精准匹配
        for vehicle_obj in annotation_list:
            if not isinstance(vehicle_obj, dict):
                continue
            if vehicle_obj.get("filename") == image_name:
                print(f" 找到{image_name}的COT标注（数组共{len(annotation_list)}个对象）")
                annot_parts = []

                # 提取reasoning_chain（增加类型校验）
                reasoning_chain = vehicle_obj.get("reasoning_chain")
                if isinstance(reasoning_chain, str):
                    reasoning_text = reasoning_chain.strip()
                    annot_parts.append(f"Reasoning: {reasoning_text}")
                else:
                    print(f" {image_name}的reasoning_chain不是字符串（类型：{type(reasoning_chain).__name__}）")

                return " ; ".join(annot_parts).strip()
        
        print(f"未找到{image_name}的COT标注（数组共{len(annotation_list)}个对象）")
        return ""

    def find_cot_description(self, annotation_list, image_name):
        """从JSON数组中直接提取 cot_description 作为综合文本描述"""
        if not isinstance(annotation_list, list):
            print(f"cot_description标注格式错误：预期数组，实际是 {type(annotation_list).__name__}（图像：{image_name}）")
            return ""

        for vehicle_obj in annotation_list:
            if not isinstance(vehicle_obj, dict):
                continue
            if vehicle_obj.get("filename") == image_name:
                cot_desc = vehicle_obj.get("cot_description", "")
                if isinstance(cot_desc, str) and cot_desc.strip():
                    return cot_desc.strip()
                print(f" {image_name}的cot_description为空或字段不存在")
                return ""

        print(f"未找到{image_name}的cot_description标注（数组共{len(annotation_list)}个对象）")
        return ""

    def _annotation_to_text(self, item, image_name):
        if not isinstance(item, dict):
            print(f"未找到{image_name}的标注")
            return ""

        annot_parts = []
        if "features" in item and isinstance(item["features"], list):
            features_text = "; ".join([
                f"{f['name']}: {f['description']}"
                for f in item["features"]
                if isinstance(f, dict) and "name" in f and "description" in f
            ])
            if features_text.strip():
                annot_parts.append(features_text)

        if "feature_relations" in item and isinstance(item["feature_relations"], list):
            relations_list = []
            for r in item["feature_relations"]:
                if not isinstance(r, dict):
                    continue
                relation_name = r.get("relation", "Unnamed relation")
                relation_desc = r.get("description", "No description")
                relation_value = r.get("reid_value", "No ReID value")
                relations_list.append(
                    f"Relation: {relation_name} - {relation_desc} (ReID value: {relation_value})"
                )
            if relations_list:
                annot_parts.append("; ".join(relations_list))

        return "; ".join([part for part in annot_parts if part.strip()]).strip()

    def _annotation_to_cot(self, item, image_name):
        if not isinstance(item, dict):
            print(f"未找到{image_name}的COT标注")
            return ""

        reasoning_chain = item.get("reasoning_chain")
        if isinstance(reasoning_chain, str):
            return f"Reasoning: {reasoning_chain.strip()}".strip()

        print(f" {image_name}的reasoning_chain不是字符串")
        return ""

    def _annotation_to_cot_description(self, item, image_name):
        if not isinstance(item, dict):
            print(f"未找到{image_name}的cot_description标注")
            return ""

        cot_desc = item.get("cot_description", "")
        if isinstance(cot_desc, str) and cot_desc.strip():
            return cot_desc.strip()

        print(f" {image_name}的cot_description为空或字段不存在")
        return ""

    def _build_image_triplets(self, vid, modal_imgs):
        """按公共文件名构建严格的三模态配对。"""
        common_names = sorted(set(modal_imgs["RGB"]) & set(modal_imgs["NI"]) & set(modal_imgs["TI"]))
        return [
            {"pair_index": None, "RGB": name, "NI": name, "TI": name}
            for name in common_names
        ]

    def _process_dir(self, dir_path, text_dir_path, relabel=False):
        """核心目录处理：修复图像名拼接，确保与JSON filename匹配"""
        # 1. 确定JSON文件名（train/test前缀）
        prefix = 'train' if 'train' in dir_path else 'test'
        suffix = self.train_text_suffix if prefix == "train" else self.test_text_suffix
        json_paths = {
            "RGB": osp.join(text_dir_path, f"{prefix}_RGB{suffix}.json"),
            "NI": osp.join(text_dir_path, f"{prefix}_NI{suffix}.json"),
            "TI": osp.join(text_dir_path, f"{prefix}_TI{suffix}.json")
        }

        # 2. 加载三模态标注
        annotations = {}
        for modal, json_path in json_paths.items():
            annotations[modal] = self._build_annotation_store(self._load_json_annotations(json_path))

        # 3. 车辆ID重标记（保持原有逻辑）
        vid_container = set()
        for vid in os.listdir(dir_path):
            vid_dir = osp.join(dir_path, vid)
            if osp.isdir(vid_dir):
                vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)} if relabel else None

        dataset = []
        cam_set = set()

        # 4. 遍历每个车辆ID目录（vid是车辆ID字符串，如"0471"）
        for vid in os.listdir(dir_path):
            vid_dir = osp.join(dir_path, vid)
            if not osp.isdir(vid_dir):
                continue

            # 三模态图像目录（WMVEID863结构：vis/RGB, ni/NI, th/TI）
            modal_dirs = {
                "RGB": osp.join(vid_dir, 'vis'),
                "NI": osp.join(vid_dir, 'ni'),
                "TI": osp.join(vid_dir, 'th')
            }

            # 检查三模态目录是否齐全
            if not all([osp.exists(d) for d in modal_dirs.values()]):
                warnings.warn(f" 车辆{vid}缺失模态目录（vis/ni/th），跳过该车辆")
                continue

            # 5. 获取三模态图像列表（按名称排序，确保一一对应）
            modal_imgs = {}
            for modal, dir_p in modal_dirs.items():
                imgs = sorted([f for f in os.listdir(dir_p) if f.endswith(('.jpg', '.png'))])
                modal_imgs[modal] = imgs

            triplets = self._build_image_triplets(vid, modal_imgs)
            if not triplets:
                warnings.warn(f" 车辆{vid}无有效图像，跳过该车辆")
                continue

            # 6. 遍历每张图像（多模态按索引匹配）
            for triplet in triplets:
                # 获取三模态图像名（原始图像名，如"v1_n1_s999_011.jpg"）
                rgb_img_raw = triplet["RGB"]
                ni_img_raw = triplet["NI"]
                ti_img_raw = triplet["TI"]
                pair_index = triplet["pair_index"]

                # 核心：拼接车辆ID前缀，形成与JSON filename一致的名称（如"0471_v1_n1_s999_011.jpg"）
                rgb_img_name = f"{vid}_{rgb_img_raw}"
                ni_img_name = f"{vid}_{ni_img_raw}"
                ti_img_name = f"{vid}_{ti_img_raw}"

                # 构建图像路径
                img_paths = (
                    osp.join(modal_dirs["RGB"], rgb_img_raw),
                    osp.join(modal_dirs["NI"], ni_img_raw),
                    osp.join(modal_dirs["TI"], ti_img_raw)
                )

                # 车辆ID和标签
                current_vid = int(vid)
                label = vid2label[current_vid] if relabel else current_vid

                # 提取摄像头ID（从原始图像名解析v字段）
                cam_match = re.search(r'v+\d', rgb_img_raw)
                night_match = re.search(r'n+\d', rgb_img_raw)
                if not cam_match or not night_match:
                    warnings.warn(f" 图像名格式异常：{rgb_img_raw}（车辆{vid}，跳过该图像）")
                    continue
                camid = int(cam_match.group(0)[1:])
                sceneid = -1  # 无场景ID时设为-1
                cam_set.add(camid)

                # 7. 用拼接后的完整名称匹配标注（与JSON的filename完全一致）
                rgb_item = self._resolve_annotation_item(annotations["RGB"], rgb_img_name, vid, pair_index)
                ni_item = self._resolve_annotation_item(annotations["NI"], ni_img_name, vid, pair_index)
                ti_item = self._resolve_annotation_item(annotations["TI"], ti_img_name, vid, pair_index)

                rgb_text = self._annotation_to_cot_description(rgb_item, rgb_img_name)
                ni_text = self._annotation_to_cot_description(ni_item, ni_img_name)
                ti_text = self._annotation_to_cot_description(ti_item, ti_img_name)

                rgb_cot = 'An image of a vehicle in the visible spectrum, capturing natural colors and fine details: ' + self._annotation_to_cot(rgb_item, rgb_img_name)
                ni_cot = 'An image of a vehicle in the near infrared spectrum, capturing contrasts and surface reflectance: ' + self._annotation_to_cot(ni_item, ni_img_name)
                ti_cot = 'An image of a vehicle in the thermal infrared spectrum, capturing heat emissions as temperature gradients: ' + self._annotation_to_cot(ti_item, ti_img_name)
                
             
                text_rgb = rgb_text
                text_ni =  ni_text
                text_ti =  ti_text

                # 9. 添加样本到数据集
                dataset.append((
                    img_paths, label, camid, sceneid,
                    text_rgb, text_ni, text_ti,
                    rgb_cot, ni_cot, ti_cot
                ))

                # print(f"rgb_text:{rgb_text}")
                # print(f"rgb_cot:{rgb_cot}")
            

        # 输出处理结果统计
        print(f" 完成{dir_path}处理：{len(dataset)}个样本，{len(cam_set)}个摄像头ID")
        return dataset

    def get_imagedata_info(self, data):
        """统计数据集信息"""
        pids, imgs, cams, vids = [], [], [], []
        for img_paths, pid, cam, vid, _, _, _, _, _, _ in data:
            pids.append(pid)
            imgs.extend(img_paths)
            cams.append(cam)
            vids.append(vid)
        return len(set(pids)), len(imgs), len(set(cams)), len(set(vids))

    def print_dataset_statistics(self, train, query, gallery):
        train_stats = self.get_imagedata_info(train)
        query_stats = self.get_imagedata_info(query)
        gallery_stats = self.get_imagedata_info(gallery)

        print(f"Dataset Statistics:")
        print(f"  Train: # PIDs: {train_stats[0]}, # Images: {train_stats[1]}, # Cameras: {train_stats[2]}")
        print(f"  Query: # PIDs: {query_stats[0]}, # Images: {query_stats[1]}, # Cameras: {query_stats[2]}")
        print(f"  Gallery: # PIDs: {gallery_stats[0]}, # Images: {gallery_stats[1]}, # Cameras: {gallery_stats[2]}")

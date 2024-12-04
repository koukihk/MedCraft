import random
from typing import Hashable, Mapping, Dict

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

from .utils import (SynthesisTumor, get_predefined_texture, get_predefined_texture_O,
                    get_predefined_texture_A, get_predefined_texture_B, get_predefined_texture_C,)

Organ_List = {'liver': [1, 2], 'pancreas': [1, 2], 'kidney': [1, 2]}
Organ_HU = {'liver': [100, 160], 'pancreas': [100, 160], 'kidney': [140, 200]}


class TumorGenerated(RandomizableTransform, MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 prob: float = 0.1,
                 tumor_prob=[0.2, 0.2, 0.2, 0.2, 0.2],
                 allow_missing_keys: bool = False,
                 gmm_list=[],
                 ellipsoid_model=None,
                 model_name=None,
                 tumor_filter=None,
                 filter_enabled: bool = False,
                 filter_threshold: float = 0.5
                 ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        random.seed(0)
        np.random.seed(0)
        self.gmm_list = gmm_list
        self.ellipsoid_model = ellipsoid_model
        self.model_name = model_name
        self.kernel_size = (3, 3, 3)  # Receptive Field
        self.organ_hu_lowerbound = Organ_HU['liver'][0]  # organ hu lowerbound
        self.outrange_standard_val = Organ_HU['liver'][1]  # outrange standard value
        self.organ_standard_val = 0  # organ standard value
        self.hu_processor = False
        self.edge_advanced_blur = True
        self.tumor_filter = tumor_filter
        self.filter_enabled = filter_enabled
        self.filter_threshold = filter_threshold

        self.tumor_types = ['tiny', 'small', 'medium', 'large', 'mix']

        assert len(tumor_prob) == 5
        self.tumor_prob = np.array(tumor_prob)
        # texture shape: 420, 300, 320
        # self.textures = pre_define 10 texture
        self.textures = []
        sigma_as = [3, 6, 9, 12, 15]
        sigma_bs = [4, 7]
        predefined_texture_shape = (420, 300, 320)
        for sigma_a in sigma_as:
            for sigma_b in sigma_bs:
                texture = get_predefined_texture_B(predefined_texture_shape, sigma_a, sigma_b)
                self.textures.append(texture)
        print("All predefined texture have generated.")

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if self._do_transform and (np.max(d['label']) <= 1):
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            texture = random.choice(self.textures)
            # 合成肿瘤
            synthesized_image, synthesized_label = SynthesisTumor(d['image'][0], d['label'][0], tumor_type, texture,
                                                          self.hu_processor, self.organ_standard_val,
                                                          self.organ_hu_lowerbound,
                                                          self.outrange_standard_val, self.edge_advanced_blur,
                                                          self.gmm_list, self.ellipsoid_model, self.model_name)

            # 如果启用过滤功能，计算合成肿瘤质量
            if self.filter_enabled and self.tumor_filter:
                quality_score = self._calculate_quality_score(synthesized_image, synthesized_label)
                if quality_score < self.filter_threshold:
                    # 未通过质量测试，放弃合成结果
                    return d

            # 更新数据
            d['image'][0], d['label'][0] = synthesized_image, synthesized_label

        return d

    def _calculate_quality_score(self, synthesized_image, synthesized_label):
        """利用过滤器计算合成肿瘤的质量得分"""
        with torch.no_grad():
            # 输入合成肿瘤，获取分割预测
            tumor_mask = torch.tensor(synthesized_label == 1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tumor_input = torch.tensor(synthesized_image).unsqueeze(0).unsqueeze(0)
            tumor_prediction = self.tumor_filter(tumor_input)

            # 计算与真实标签的匹配比例 (公式4)
            predicted_tumor = (tumor_prediction > 0.5).float()
            matching_voxels = (predicted_tumor * tumor_mask).sum().item()
            total_tumor_voxels = tumor_mask.sum().item()
            quality_score = matching_voxels / (total_tumor_voxels + 1e-8)  # 避免除零

        return quality_score
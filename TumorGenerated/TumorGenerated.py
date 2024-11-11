import random
from typing import Hashable, Mapping, Dict

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

from .utils import (SynthesisTumor, get_predefined_texture, get_predefined_texture_O,
                    get_predefined_texture_A, get_predefined_texture_B, get_predefined_texture_C,)
from .filter import SyntheticTumorFilter

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
                 segmentor=None,
                 filter_threshold: float = 0.5,
                 filter_enabled: bool = False
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
        self.edge_advanced_blur = False

        self.tumor_types = ['tiny', 'small', 'medium', 'large', 'mix']

        assert len(tumor_prob) == 5
        self.tumor_prob = np.array(tumor_prob)

        # 预定义纹理生成
        self.textures = []
        sigma_as = [3, 6, 9, 12, 15]
        sigma_bs = [4, 7]
        predefined_texture_shape = (420, 300, 320)
        for sigma_a in sigma_as:
            for sigma_b in sigma_bs:
                texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
                self.textures.append(texture)
        print("All predefined texture have generated.")

        # 初始化过滤器
        self.filter_enabled = filter_enabled
        if self.filter_enabled and segmentor is not None:
            self.tumor_filter = SyntheticTumorFilter(segmentor, filter_threshold)
        else:
            self.tumor_filter = None

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if self._do_transform and (np.max(d['label']) <= 1):
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            texture = random.choice(self.textures)

            # 生成肿瘤
            synthesized_image, synthesized_label = SynthesisTumor(
                d['image'][0], d['label'][0], tumor_type, texture,
                self.hu_processor, self.organ_standard_val,
                self.organ_hu_lowerbound, self.outrange_standard_val,
                self.edge_advanced_blur, self.gmm_list,
                self.ellipsoid_model, self.model_name
            )

            # 如果启用了过滤器，则进行质量检测
            if self.filter_enabled and self.tumor_filter is not None:
                passed_quality_check = self.tumor_filter.filter(synthesized_image, synthesized_label)
                if passed_quality_check:
                    d['image'][0], d['label'][0] = synthesized_image, synthesized_label
                else:
                    print("Generated tumor did not pass quality check, reverting to original sample.")
            else:
                d['image'][0], d['label'][0] = synthesized_image, synthesized_label

        return d

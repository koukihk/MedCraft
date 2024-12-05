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
                 filter_model=None,
                 filter_inferer=None,
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
        self.filter_model = filter_model
        self.filter_inferer = filter_inferer
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
            synthesized_image, synthesized_label = SynthesisTumor(
                d['image'][0], d['label'][0], tumor_type, texture,
                self.hu_processor, self.organ_standard_val,
                self.organ_hu_lowerbound, self.outrange_standard_val,
                self.edge_advanced_blur, self.gmm_list,
                self.ellipsoid_model, self.model_name
            )

            if self.filter_enabled and self.filter_model is not None and self.filter_inferer is not None:
                with torch.no_grad():
                    # 将 synthesized_image 从 numpy.ndarray 转换为 tensor
                    synthesized_image = torch.from_numpy(synthesized_image).float()

                    # 通过过滤器模型进行质量检测
                    if self.filter_inferer is not None:
                        filter_logits = self.filter_inferer(synthesized_image.unsqueeze(0).unsqueeze(0).cuda())
                    else:
                        filter_logits = self.filter_model(synthesized_image.unsqueeze(0).unsqueeze(0).cuda())
                    filter_probs = torch.softmax(filter_logits, dim=1).cpu().numpy()
                    filter_mask = np.argmax(filter_probs, axis=1).astype(np.uint8)

                    # 根据过滤器的输出计算满意比例
                    M = synthesized_label == 2  # 真实肿瘤掩码
                    S = filter_mask[0] == 2  # 过滤器分割结果
                    P = np.sum(S & M) / np.sum(M) if np.sum(M) > 0 else 0

                    if P < self.filter_threshold:
                        # 不满足质量要求，返回原始数据
                        print(f"Synthetic tumor discarded. Quality score: {P:.4f}")
                        return d

            # 如果通过了过滤，更新样本数据
            d['image'][0] = synthesized_image
            d['label'][0] = synthesized_label

        return d

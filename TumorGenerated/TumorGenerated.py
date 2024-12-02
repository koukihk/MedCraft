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
                 tumor_filter=None,
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
        self.edge_advanced_blur = True

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

        # 初始化3D肿瘤过滤器
        self.filter_enabled = filter_enabled
        if self.filter_enabled and tumor_filter is not None:
            self.tumor_filter = tumor_filter

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if self._do_transform and (np.max(d['label']) <= 1):
            # 保存原始3D图像和标签
            if isinstance(d['image'], torch.Tensor):
                original_image = d['image'].clone()
                original_label = d['label'].clone()
            else:
                original_image = d['image'].copy()  # numpy array
                original_label = d['label'].copy()  # numpy array

            # 确保维度正确
            if isinstance(original_image, np.ndarray):
                if original_image.ndim == 3:
                    original_image = original_image[np.newaxis, ...]  # 添加通道维度
                if original_label.ndim == 3:
                    original_label = original_label[np.newaxis, ...]

            # 生成3D合成肿瘤
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            texture = random.choice(self.textures)

            # 获取第一个通道的数据
            img_data = original_image[0] if original_image.ndim == 4 else original_image
            label_data = original_label[0] if original_label.ndim == 4 else original_label

            synthetic_image, synthetic_label = SynthesisTumor(
                img_data, label_data, tumor_type, texture,
                self.hu_processor, self.organ_standard_val,
                self.organ_hu_lowerbound, self.outrange_standard_val,
                self.edge_advanced_blur, self.gmm_list,
                self.ellipsoid_model, self.model_name
            )

            # 如果启用过滤器，进行3D质量检查
            if self.filter_enabled and self.tumor_filter is not None:
                synthetic_image = torch.as_tensor(synthetic_image, dtype=torch.float32, device='cpu')
                synthetic_label = torch.as_tensor(synthetic_label, dtype=torch.float32, device='cpu')

                passed = self.tumor_filter(synthetic_image, synthetic_label)
                if not passed:
                    # 质量检测未通过，使用原始3D图像
                    synthetic_image = img_data
                    synthetic_label = label_data

            # 更新数据字典
            if isinstance(d['image'], torch.Tensor):
                d['image'][0] = torch.as_tensor(synthetic_image)
                d['label'][0] = torch.as_tensor(synthetic_label)
            else:
                d['image'][0] = np.asarray(synthetic_image)
                d['label'][0] = np.asarray(synthetic_label)

        return d

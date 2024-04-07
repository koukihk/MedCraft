import os
import random
from typing import Hashable, Mapping, Dict

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, RandomizableTransform

from .utils import SynthesisTumor, get_predefined_texture
import numpy as np

import nibabel as nib


class TumorGenerated(RandomizableTransform, MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 prob: float = 0.1,
                 tumor_prob=[0.2, 0.2, 0.2, 0.2, 0.2],
                 allow_missing_keys: bool = False,
                 save_flag: bool = False,
                 gmm_model=None
                 ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.save_flag = save_flag
        self.gmm_model = gmm_model
        random.seed(0)
        np.random.seed(0)

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
                texture = get_predefined_texture(predefined_texture_shape, sigma_a, sigma_b)
                self.textures.append(texture)
        print("All predefined texture have generated.")

    def save_data(self, d, tumor_type):
        def get_datatype(datatype):
            data_type_map = {
                2: 'uint8',
                4: 'int16',
                8: 'int32',
                16: 'float32',
                32: 'complex64',
                64: 'float64'
            }
            return data_type_map.get(datatype, 'uint8')

        image_data_type = get_datatype(d['image_meta_dict']['datatype'][()])
        image_affine_matrix = d['image_meta_dict']['original_affine']

        label_data_type = get_datatype(d['label_meta_dict']['datatype'][()])
        label_affine_matrix = d['label_meta_dict']['original_affine']

        image = d['image'][0]
        label = d['label'][0]

        image_outputs = f'synt/{tumor_type}/image'
        label_outputs = f'synt/{tumor_type}/label'

        image_filename = os.path.basename(d['image_meta_dict']['filename_or_obj']).split('/')[-1]
        label_filename = os.path.basename(d['label_meta_dict']['filename_or_obj']).split('/')[-1]

        os.makedirs(image_outputs, exist_ok=True)
        os.makedirs(label_outputs, exist_ok=True)

        nib.save(
            nib.Nifti1Image(image.astype(image_data_type), image_affine_matrix, header=d['image_meta_dict']),
            os.path.join(image_outputs, f'synt_{image_filename}')
        )

        nib.save(
            nib.Nifti1Image(label.astype(label_data_type), label_affine_matrix, header=d['label_meta_dict']),
            os.path.join(label_outputs, f'synt_{label_filename}')
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        if self._do_transform and (np.max(d['label']) <= 1):
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_prob.ravel())
            texture = random.choice(self.textures)
            d['image'][0], d['label'][0] = SynthesisTumor(d['image'][0], d['label'][0], tumor_type, texture, self.gmm_model)
            # print(tumor_type, d['image'].shape, np.max(d['label']))
            if self.save_flag:
                self.save_data(d, tumor_type)
        return d
import torch
import numpy as np
from monai.transforms import MapTransform

class Mixup3D(MapTransform):
    def __init__(self, keys, alpha=0.4, prob=0.5):
        super().__init__(keys)
        self.alpha = alpha
        self.prob = prob

    def __call__(self, data):
        lam = np.random.beta(self.alpha, self.alpha)
        if np.random.rand() > self.prob:
            return data

        image1, label1 = data['image'], data['label']
        image2, label2 = data['mix_image'], data['mix_label']

        image_mix = lam * image1 + (1 - lam) * image2

        label1_onehot = torch.nn.functional.one_hot(label1.long(), num_classes=3).float()
        label2_onehot = torch.nn.functional.one_hot(label2.long(), num_classes=3).float()
        label_mix_onehot = lam * label1_onehot + (1 - lam) * label2_onehot
        label_mix = torch.argmax(label_mix_onehot, dim=-1)

        data['image'] = image_mix
        data['label'] = label_mix
        return data

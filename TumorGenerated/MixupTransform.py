import random

import numpy as np
import torch
from monai import transforms


class MixupTransform(transforms.Transform):
    def __init__(self, keys=["image", "label"], alpha=0.5, prob=0.5):
        super().__init__(keys)
        self.alpha = alpha
        self.prob = prob

    def __call__(self, data):
        if random.random() > self.prob:
            return data

        d = dict(data)
        lam = np.random.beta(self.alpha, self.alpha)

        if "image" in self.keys:
            image_a = d["image"]
            image_b = torch.flip(image_a, [0])  # 使用batch内的其他样本
            d["image"] = lam * image_a + (1 - lam) * image_b

        if "label" in self.keys:
            label_a = d["label"]
            label_b = torch.flip(label_a, [0])
            d["label"] = lam * label_a + (1 - lam) * label_b

        d["mixup_lambda"] = lam
        return d
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

        image1 = np.load(data['image'][0]) if isinstance(data['image'][0], str) else data['image'][0]
        label1 = np.load(data['label'][0]) if isinstance(data['label'][0], str) else data['label'][0]
        image2 = np.load(data['mix_image'][0]) if isinstance(data['mix_image'][0], str) else data['mix_image'][0]
        label2 = np.load(data['mix_label'][0]) if isinstance(data['mix_label'][0], str) else data['mix_label'][0]

        image1 = torch.tensor(image1) if not isinstance(image1, torch.Tensor) else image1
        image2 = torch.tensor(image2) if not isinstance(image2, torch.Tensor) else image2

        image_mix = lam * image1.float() + (1 - lam) * image2.float()

        label1_onehot = torch.nn.functional.one_hot(label1.long(), num_classes=3).float()
        label2_onehot = torch.nn.functional.one_hot(label2.long(), num_classes=3).float()
        label_mix_onehot = lam * label1_onehot + (1 - lam) * label2_onehot
        label_mix = torch.argmax(label_mix_onehot, dim=-1)

        data['image'][0] = image_mix
        data['label'][0] = label_mix
        return data

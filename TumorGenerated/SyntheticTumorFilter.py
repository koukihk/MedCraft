import torch
import torch.nn.functional as F


class SyntheticTumorFilter:
    def __init__(self, model, inferer, device, threshold=0.5):
        self.model = model
        self.inferer = inferer
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def calculate_quality_score(self, image, mask):
        if image.ndim == 3:  # [D,H,W]
            image = image.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        elif image.ndim == 4:  # [C,D,H,W]
            image = image.unsqueeze(0)  # [1,C,D,H,W]

        if mask.ndim == 3:  # [D,H,W]
            mask = mask.unsqueeze(0)  # [1,D,H,W]

        image = image.to(self.device, dtype=torch.float32)
        mask = mask.to(self.device, dtype=torch.float32)

        pred = self.inferer(image)  # [B,C,D,H,W]
        pred = F.softmax(pred, dim=1)
        pred = (pred.argmax(dim=1) > 0).float()  # [B,D,H,W]

        tumor_mask = (mask > 0)
        intersection = torch.sum(pred * tumor_mask)
        total_tumor_voxels = torch.sum(tumor_mask)

        if total_tumor_voxels == 0:
            return 0.0

        quality_score = (intersection / total_tumor_voxels).cpu().item()
        return quality_score

    def __call__(self, image, mask):
        quality_score = self.calculate_quality_score(image, mask)
        return quality_score >= self.threshold
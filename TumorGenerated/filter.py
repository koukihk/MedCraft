import torch
import numpy as np


class SyntheticTumorFilter:
    def __init__(self, segmentor, threshold=0.5):
        """Initialize the synthetic tumor filter

        Args:
            segmentor: Trained segmentation model
            threshold: Quality threshold T (default: 0.5)
        """
        self.segmentor = segmentor
        self.threshold = threshold
        self.segmentor.eval()  # Set model to evaluation mode

    def calculate_proportion(self, synthetic_image, tumor_mask):
        """Calculate the proportion P between synthetic tumor and mask

        Args:
            synthetic_image: Synthesized image tensor (C, H, W, D)
            tumor_mask: Tumor mask tensor (H, W, D)
        Returns:
            float: Proportion P
        """
        # Ensure input is on GPU and has correct dimensions
        if not torch.is_tensor(synthetic_image):
            synthetic_image = torch.from_numpy(synthetic_image)
        if not torch.is_tensor(tumor_mask):
            tumor_mask = torch.from_numpy(tumor_mask)

        if len(synthetic_image.shape) == 3:
            synthetic_image = synthetic_image.unsqueeze(0)

        with torch.no_grad():
            pred = self.segmentor(synthetic_image.unsqueeze(0).cuda())
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        pred_tumor = (pred == 2).astype(np.int32)
        tumor_mask = (tumor_mask == 1).astype(np.int32)

        numerator = np.sum(pred_tumor * tumor_mask)
        denominator = np.sum(tumor_mask)

        if denominator == 0:
            return 0.0

        proportion = numerator / denominator
        return proportion

    def filter(self, synthetic_image, tumor_mask):
        """Check if synthetic tumor passes the quality test

        Args:
            synthetic_image: Synthesized image tensor
            tumor_mask: Tumor mask tensor
        Returns:
            bool: Whether passes the quality test
        """
        proportion = self.calculate_proportion(synthetic_image, tumor_mask)
        return proportion >= self.threshold

    def batch_filter(self, synthetic_images, tumor_masks):
        """Filter a batch of synthetic tumors

        Args:
            synthetic_images: Batch of synthesized images
            tumor_masks: Batch of tumor masks
        Returns:
            list: List of boolean values indicating which images passed the test
        """
        return [self.filter(img, mask) for img, mask in zip(synthetic_images, tumor_masks)]


def create_filter(segmentor, threshold=0.5):
    """Factory function to create a tumor filter

    Args:
        segmentor: Trained segmentation model
        threshold: Quality threshold T
    Returns:
        SyntheticTumorFilter: Configured filter instance
    """
    return SyntheticTumorFilter(segmentor, threshold)
import torch
import numpy as np


class SyntheticTumorFilter:
    def __init__(self, segmentor, threshold=0.5):
        """Initialize the synthetic tumor filter

        Args:
            segmentor: Trained segmentation model
            threshold: Quality threshold T (default: 0.5)
        """
        self.segmentor = segmentor.cpu()  # 将模型放在CPU
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
        if not torch.is_tensor(synthetic_image):
            synthetic_image = torch.from_numpy(synthetic_image)
        if not torch.is_tensor(tumor_mask):
            tumor_mask = torch.from_numpy(tumor_mask)

        # 确保数据类型为float
        synthetic_image = synthetic_image.float()

        # 标准化输入维度
        if len(synthetic_image.shape) == 3:  # (H, W, D)
            synthetic_image = synthetic_image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
        elif len(synthetic_image.shape) == 4:  # (C, H, W, D)
            synthetic_image = synthetic_image.unsqueeze(0)  # (1, C, H, W, D)

        # 添加调试信息
        print("Input shape to segmentor:", synthetic_image.shape)

        with torch.no_grad():
            try:
                # 确保输入大小是模型期望的大小
                expected_size = (256, 256, 256)  # 替换为你的模型期望的输入大小
                if synthetic_image.shape[2:] != expected_size:
                    synthetic_image = torch.nn.functional.interpolate(
                        synthetic_image,
                        size=expected_size,
                        mode='trilinear',
                        align_corners=False
                    )

                pred = self.segmentor(synthetic_image)
                pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

            except Exception as e:
                print(f"Error during segmentation: {e}")
                print(f"Input tensor shape: {synthetic_image.shape}")
                raise

        pred_tumor = (pred == 2).astype(np.int32)
        tumor_mask = (tumor_mask == 1).astype(np.int32)

        # 确保 pred_tumor 和 tumor_mask 具有相同的形状
        if pred_tumor.shape != tumor_mask.shape:
            print(f"Shape mismatch - pred_tumor: {pred_tumor.shape}, tumor_mask: {tumor_mask.shape}")
            # 将预测调整为与mask相同的大小
            tumor_mask = torch.nn.functional.interpolate(
                torch.from_numpy(tumor_mask).float().unsqueeze(0).unsqueeze(0),
                size=pred_tumor.shape,
                mode='nearest'
            ).squeeze().numpy()

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
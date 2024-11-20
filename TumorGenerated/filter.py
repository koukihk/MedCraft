import torch
import torch.nn.functional as F


class SyntheticTumorFilter:
    def __init__(self, model, inferer, threshold=0.5):
        """
        Args:
            model: 预训练的3D分割模型
            inferer: sliding window inference函数，用于大体积3D数据的推理
            threshold: 质量评估阈值
        """
        self.model = model
        self.inferer = inferer
        self.threshold = threshold

    @torch.no_grad()
    def calculate_quality_score(self, image, mask):
        """
        计算3D合成肿瘤的质量分数
        Args:
            image: 3D图像 shape: [C,D,H,W] 或 [D,H,W]
            mask: 3D肿瘤掩码 shape: [C,D,H,W] 或 [D,H,W]
        Returns:
            quality_score: 质量分数
        """
        # 标准化输入维度
        if image.ndim == 3:  # [D,H,W]
            image = image.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        elif image.ndim == 4:  # [C,D,H,W]
            image = image.unsqueeze(0)  # [1,C,D,H,W]

        if mask.ndim == 3:  # [D,H,W]
            mask = mask.unsqueeze(0)  # [1,D,H,W]

        # 确保数据类型正确
        image = image.float().cuda()
        mask = mask.float().cuda()

        # 使用sliding window inference进行3D分割预测
        # pred shape: [B,C,D,H,W]
        pred = self.inferer(image)
        pred = F.softmax(pred, dim=1)
        pred = (pred.argmax(dim=1) > 0).float()  # [B,D,H,W]

        # 计算重叠比例
        tumor_mask = (mask > 0)
        intersection = torch.sum(pred * tumor_mask)
        total_tumor_voxels = torch.sum(tumor_mask)

        if total_tumor_voxels == 0:
            return 0.0

        quality_score = (intersection / total_tumor_voxels).cpu().item()
        return quality_score

    def __call__(self, image, mask):
        """
        执行3D肿瘤质量过滤
        Args:
            image: 3D图像
            mask: 3D肿瘤掩码
        Returns:
            passed: 是否通过质量检测
        """
        quality_score = self.calculate_quality_score(image, mask)
        return quality_score >= self.threshold
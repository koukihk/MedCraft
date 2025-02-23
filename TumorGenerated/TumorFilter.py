import numpy as np
import torch
from monai.transforms import Transform
from scipy.ndimage import label, binary_dilation


class TumorFilter(Transform):
    def __init__(self, model, model_inferer, use_inferer=True, threshold=0.5):
        self.model = model
        self.model_inferer = model_inferer
        self.use_inferer = use_inferer
        self.threshold = threshold

    def __call__(self, data):
        image = data['image']
        label_data = data['label']

        filtered_image, filtered_label = filter_synthetic_tumor_batch(
            image, label_data, self.model, self.model_inferer, self.use_inferer, self.threshold
        )

        if filtered_image is None or filtered_label is None:
            raise ValueError("No high-quality synthetic tumors passed the filtering process.")

        data['image'] = filtered_image
        data['label'] = filtered_label
        return data


def denoise_pred(pred: np.ndarray):
    if pred.ndim == 4:
        return denoise_single_sample(pred)
    elif pred.ndim == 5:
        batch_denoised = np.zeros_like(pred)
        for i in range(pred.shape[0]):
            batch_denoised[i, ...] = denoise_single_sample(pred[i, ...])
        return batch_denoised
    else:
        raise ValueError(f"Unexpected input dimensions: {pred.shape}")


def denoise_single_sample(pred: np.ndarray):
    denoise_pred = np.zeros_like(pred)

    live_channel = pred[1, ...]
    labels, nb = label(live_channel)
    max_sum = -1
    choice_idx = -1
    for idx in range(1, nb + 1):
        component = (labels == idx)
        if np.sum(component) > max_sum:
            choice_idx = idx
            max_sum = np.sum(component)
    component = (labels == choice_idx)
    denoise_pred[1, ...] = component

    liver_dilation = binary_dilation(denoise_pred[1, ...], iterations=30).astype(bool)
    denoise_pred[2, ...] = pred[2, ...].astype(bool) * liver_dilation

    denoise_pred[0, ...] = 1 - np.logical_or(denoise_pred[1, ...], denoise_pred[2, ...])
    return denoise_pred


def calculate_quality_proportion(segmentation_output, tumor_mask):
    # tumor_mask 可能为 torch.Tensor，因此调用 .float()
    tumor_mask = (tumor_mask == 2).float()
    tumor_voxels = tumor_mask.sum().item()
    if tumor_voxels == 0:
        return 0  # 避免除以0
    if segmentation_output.ndim == 5:
        seg_tumor_prob = segmentation_output[:, 2, ...]
    else:
        seg_tumor_prob = segmentation_output[2, ...]
    matched_voxels = (seg_tumor_prob * tumor_mask).sum().item()
    return matched_voxels / tumor_voxels


def filter_synthetic_tumor_batch(data, target, model, model_inferer, use_inferer, threshold=0.5):
    # 记录输入数据类型，如果为 numpy，则处理完后转换回 numpy
    input_type = 'tensor'
    if isinstance(data, np.ndarray):
        input_type = 'numpy'
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)

    filtered_data = []
    filtered_target = []
    num_samples = data.size(0)  # 此时 data 是 torch.Tensor

    for i in range(num_samples):
        single_data = data[i].unsqueeze(0)
        single_target = target[i].unsqueeze(0)

        with torch.no_grad():
            output = model_inferer(single_data) if use_inferer else model(single_data)
            # 将输出转换为 numpy 数组进行去噪处理
            output_np = output.detach().cpu().numpy()
            output_np = denoise_pred(output_np)
            # 再转换为 torch.Tensor
            output = torch.tensor(output_np, device=single_data.device)

            quality_proportion = calculate_quality_proportion(output, single_target)
            if quality_proportion >= threshold:
                filtered_data.append(single_data)
                filtered_target.append(single_target)

    if len(filtered_data) == 0:
        return None, None

    filtered_data = torch.cat(filtered_data, dim=0)
    filtered_target = torch.cat(filtered_target, dim=0)

    # 如果输入是 numpy 数组，则转换回 numpy 格式
    if input_type == 'numpy':
        filtered_data = filtered_data.detach().cpu().numpy()
        filtered_target = filtered_target.detach().cpu().numpy()

    return filtered_data, filtered_target

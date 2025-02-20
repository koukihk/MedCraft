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
        label = data['label']

        # Assuming the model outputs are logits and need post-processing
        filtered_data, filtered_target = filter_synthetic_tumor_batch(
            image, label, self.model, self.model_inferer, self.use_inferer, self.threshold
        )

        if filtered_data is None or filtered_target is None:
            raise ValueError("No high-quality synthetic tumors passed the filtering process.")

        # Return filtered data and target in the same format as the input dictionary
        data['image'] = filtered_data
        data['label'] = filtered_target

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
    tumor_mask = (tumor_mask == 2).float()
    tumor_voxels = tumor_mask.sum().item()
    if tumor_voxels == 0:
        return 0  # Avoid division by zero
    if segmentation_output.ndim == 5:
        seg_tumor_prob = segmentation_output[:, 2, ...]
    else:
        seg_tumor_prob = segmentation_output[2, ...]
    matched_voxels = (seg_tumor_prob * tumor_mask).sum().item()
    return matched_voxels / tumor_voxels


def filter_synthetic_tumor_batch(data, target, model, model_inferer, use_inferer, threshold=0.5):
    filtered_data = []
    filtered_target = []

    for i in range(data.size(0)):
        single_data = data[i].unsqueeze(0)
        single_target = target[i].unsqueeze(0)

        with torch.no_grad():
            output = model_inferer(single_data) if use_inferer else model(single_data)
            output = denoise_pred(output.detach().cpu().numpy())
            output = torch.tensor(output, device=data.device)

            quality_proportion = calculate_quality_proportion(output, single_target)
            if quality_proportion >= threshold:
                filtered_data.append(single_data)
                filtered_target.append(single_target)

    if len(filtered_data) == 0:
        return None, None
    return torch.cat(filtered_data, dim=0), torch.cat(filtered_target, dim=0)

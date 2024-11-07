import numpy as np


def calculate_proportion(segmentation_result, tumor_mask):
    """
    Calculate the proportion of satisfactory synthesized tumor regions

    Args:
        segmentation_result: Binary numpy array from segmentor output
        tumor_mask: Binary numpy array of ground truth tumor mask

    Returns:
        float: Proportion P as defined in equation (4)
    """
    # Convert inputs to binary arrays if they aren't already
    seg_binary = segmentation_result > 0
    mask_binary = tumor_mask > 0

    # Calculate numerator (intersection of segmented tumor and mask)
    numerator = np.sum(np.logical_and(seg_binary, mask_binary))

    # Calculate denominator (total number of tumor mask voxels)
    denominator = np.sum(mask_binary)

    # Avoid division by zero
    if denominator == 0:
        return 0.0

    return numerator / denominator


def quality_filter(synthetic_image, tumor_mask, segmentor, threshold=0.5):
    """
    Filter synthetic tumors based on segmentation quality

    Args:
        synthetic_image: The synthesized image with tumor
        tumor_mask: Binary mask showing tumor locations
        segmentor: Function that performs tumor segmentation
        threshold: Quality threshold T

    Returns:
        tuple: (original_image, bool) where bool indicates if image passed quality test
    """
    # Get segmentation result from segmentor
    segmentation_result = segmentor(synthetic_image)

    # Calculate proportion P
    proportion = calculate_proportion(segmentation_result, tumor_mask)

    # Apply filtering strategy as defined in equation (5)
    passed_quality_test = proportion >= threshold

    return synthetic_image, passed_quality_test


# Example usage with dummy segmentor
def dummy_segmentor(image):
    """Dummy segmentor for demonstration"""
    return np.random.binomial(1, 0.5, size=image.shape)


# Example
if __name__ == "__main__":
    # Create dummy data
    image_shape = (64, 64, 64)
    synthetic_image = np.random.random(image_shape)
    tumor_mask = np.zeros(image_shape)
    tumor_mask[20:40, 20:40, 20:40] = 1  # Simulated tumor region

    # Apply filter
    filtered_image, passed = quality_filter(
        synthetic_image,
        tumor_mask,
        dummy_segmentor
    )

    print(f"Image passed quality test: {passed}")
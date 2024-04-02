import os
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import ndimage
import nibabel as nib

all_tumor_positions = None
gmm_model = None
has_fitted_gmm = False

def load_data_and_fit_gmm(data_folder, optimal_components):
    global all_tumor_positions, gmm_model, has_fitted_gmm

    if not has_fitted_gmm:
        if all_tumor_positions is None:
            ct_files = sorted(os.listdir(os.path.join(data_folder, "img")))
            all_tumor_positions = []

            for ct_file in ct_files:
                if ct_file.startswith("._"):
                    continue
                img_path = os.path.join(data_folder, "img", ct_file)
                label_path = os.path.join(data_folder, "label", ct_file)

                if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
                    continue

                label_data = nib.load(label_path).get_fdata()
                tumor_positions = analyze_tumor_location(label_data, 2)

                if tumor_positions:
                    all_tumor_positions.extend(tumor_positions)

            all_tumor_positions = np.array(all_tumor_positions)

        gmm_model = GaussianMixture(n_components=optimal_components)
        gmm_model.fit(all_tumor_positions)
        has_fitted_gmm = True

def get_all_tumor_positions():
    return all_tumor_positions

def get_gmm_model():
    return gmm_model

def analyze_tumor_location(label_data, tumor_label=2):
    """
    Analyze tumor location in label data.

    Parameters:
        label_data (ndarray): Label data containing tumor labels.
        tumor_label (int): Label value indicating tumor region. Default is 2.

    Returns:
        list: List of tumor positions, each position represented as a tuple (z, y, x).
    """
    labeled_components, num_components = ndimage.label(label_data == tumor_label)

    tumor_positions = []

    for i in range(1, num_components + 1):
        labeled_tumor = labeled_components == i
        tumor_indices = np.transpose(np.nonzero(labeled_tumor))

        # Calculate tumor centroid
        centroid = tuple(np.mean(tumor_indices, axis=0).astype(int))
        tumor_positions.append(centroid)

    return tumor_positions

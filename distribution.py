import os
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import ndimage
import nibabel as nib

all_tumor_positions = None
gmm_model = None

def load_data_and_fit_gmm(data_folder):
    global all_tumor_positions, gmm_model

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
            labeled_components, tumor_positions = analyze_tumor_location(label_data)

            if tumor_positions:
                all_tumor_positions.extend(tumor_positions)

        all_tumor_positions = np.array(all_tumor_positions)

    if gmm_model is None:
        gmm_model = GaussianMixture(n_components=3)
        gmm_model.fit(all_tumor_positions)

def get_all_tumor_positions():
    return all_tumor_positions

def get_gmm_model():
    return gmm_model

def analyze_tumor_location(label_data):
    labeled_components, num_components = ndimage.label(label_data == 2)
    np.save("labeled_components.npy", labeled_components)

    tumor_positions = []

    for i in range(1, num_components + 1):
        bounding_box = ndimage.find_objects(labeled_components == i)[0]
        center_coords = [int((slice.start + slice.stop - 1) / 2) for slice in bounding_box]
        tumor_positions.append(center_coords)  # 改为不使用tuple，只保留坐标数组

    return labeled_components, tumor_positions

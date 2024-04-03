import os
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import ndimage
import nibabel as nib

class TumorAnalyzer:
    def __init__(self):
        self.all_tumor_positions = None
        self.gmm_model = None
        self.has_fitted_gmm = False
        self.indices_to_skip = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]

    def fit_gmm_model(self, data, optimal_components):
        self.gmm_model = GaussianMixture(n_components=optimal_components, covariance_type='full', init_params='random', tol=0.001, max_iter=100)
        self.gmm_model.fit(data)

    def load_data(self, data_folder):
        ct_files = sorted(os.listdir(os.path.join(data_folder, "img")))
        tumor_positions = []

        for ct_file in ct_files:
            if ct_file.startswith("._"):
                continue
            img_path = os.path.join(data_folder, "img", ct_file)
            label_path = os.path.join(data_folder, "label", ct_file)

            file_index = int(ct_file.split('_')[1].split('.')[0])
            if file_index in self.indices_to_skip:
                continue

            if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
                continue

            label_data = nib.load(label_path).get_fdata()
            positions = self.analyze_tumor_location(label_data, tumor_label=2)

            if positions:
                tumor_positions.extend(positions)

        self.all_tumor_positions = np.array(tumor_positions)

    def load_data_and_fit_gmm(self, data_folder, optimal_components):
        if not self.has_fitted_gmm:
            self.load_data(data_folder)
            self.fit_gmm_model(self.all_tumor_positions, optimal_components)
            self.has_fitted_gmm = True

    def analyze_tumor_location(self, label_data, tumor_label=2):
        labeled_components, num_components = ndimage.label(label_data == tumor_label)
        tumor_positions = []

        for i in range(1, num_components + 1):
            labeled_tumor = labeled_components == i
            tumor_indices = np.transpose(np.nonzero(labeled_tumor))
            centroid = tuple(np.mean(tumor_indices, axis=0).astype(int))
            tumor_positions.append(centroid)

        return tumor_positions

    def get_all_tumor_positions(self):
        return self.all_tumor_positions

    def get_gmm_model(self):
        return self.gmm_model

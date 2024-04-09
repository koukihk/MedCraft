import os
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import ndimage
import nibabel as nib
from multiprocessing import Pool


class TumorAnalyzer:
    def __init__(self):
        self.all_tumor_positions = None
        self.gmm_model = None
        self.has_fitted_gmm = False
        self.indices_to_skip = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]

    def fit_gmm_model(self, data, optimal_components):
        """
        Fits a Gaussian Mixture Model to the given data.
        """
        self.gmm_model = GaussianMixture(n_components=optimal_components, covariance_type='full', init_params='kmeans', tol=0.0001, max_iter=200)
        self.gmm_model.fit(data)

    @staticmethod
    def process_file(ct_file, data_folder, indices_to_skip):
        if ct_file.startswith("._"):
            return []

        img_path = os.path.join(data_folder, "img", ct_file)
        label_path = os.path.join(data_folder, "label", ct_file)

        file_index = int(ct_file.split('_')[1].split('.')[0])
        if file_index in indices_to_skip:
            return []

        if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
            return []

        label_data = nib.load(label_path).get_fdata()
        positions = TumorAnalyzer.analyze_tumor_location(label_data, 1, 2)

        return positions

    def load_data(self, data_folder):
        """
        Loads CT scan images and corresponding tumor labels from the specified data folder.
        """
        ct_files = sorted(os.listdir(os.path.join(data_folder, "img")))

        with Pool() as pool:
            results = pool.starmap(TumorAnalyzer.process_file, [(ct_file, data_folder, self.indices_to_skip) for ct_file in ct_files])

        tumor_positions = [position for sublist in results for position in sublist]

        self.all_tumor_positions = np.array(tumor_positions)

    def load_data_and_fit_gmm(self, data_folder, optimal_components):
        """
        Loads data and fits GMM model.
        """
        if not self.has_fitted_gmm:
            self.load_data(data_folder)
            self.fit_gmm_model(self.all_tumor_positions, optimal_components)
            self.has_fitted_gmm = True

    @staticmethod
    def analyze_tumor_location(label_data, liver_label=1, tumor_label=2):
        """
        Analyzes tumor location from label data.
        """
        # Find liver region
        liver_mask = (label_data == liver_label)

        # Find tumor region
        tumor_mask = (label_data == tumor_label)

        # Use liver region as a mask for tumor region
        tumor_mask_in_liver = tumor_mask & liver_mask

        # Label tumor components within liver region
        labeled_tumors, num_tumors = ndimage.label(tumor_mask_in_liver)

        tumor_positions = []

        for i in range(1, num_tumors + 1):
            labeled_tumor = labeled_tumors == i
            tumor_indices = np.transpose(np.nonzero(labeled_tumor))
            centroid = tuple(np.mean(tumor_indices, axis=0))
            tumor_positions.append(centroid)

        return tumor_positions

    def get_all_tumor_positions(self):
        """
        Returns all tumor positions.
        """
        return self.all_tumor_positions

    def get_gmm_model(self):
        """
        Returns the trained GMM model.
        """
        return self.gmm_model

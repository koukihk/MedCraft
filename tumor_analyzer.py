import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import label
from scipy.optimize import least_squares
from sklearn.mixture import GaussianMixture


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
        try:
            self.gmm_model = GaussianMixture(n_components=optimal_components, covariance_type='full', init_params='kmeans', tol=0.0001, max_iter=200)
            self.gmm_model.fit(data)
        except Exception as e:
            print("Error occurred while fitting GMM model:", e)

    @staticmethod
    def process_file(ct_file, data_folder, indices_to_skip):
        try:
            if ct_file.startswith("._"):
                return []

            img_path = os.path.join(data_folder, "img", ct_file)
            label_path = os.path.join(data_folder, "label", ct_file)

            file_index = int(ct_file.split('_')[1].split('.')[0])
            if file_index in indices_to_skip:
                return []

            if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
                return []

            label = nib.load(label_path)
            positions = TumorAnalyzer.analyze_tumor_location(label, 1, 2)

            return positions
        except Exception as e:
            print("Error occurred while processing file", ct_file, ":", e)
            return []

    def load_data(self, data_folder):
        """
        Loads CT scan images and corresponding tumor labels from the specified data folder.
        """
        try:
            ct_files = sorted(os.listdir(os.path.join(data_folder, "img")))

            with Pool() as pool:
                results = pool.starmap(TumorAnalyzer.process_file, [(ct_file, data_folder, self.indices_to_skip) for ct_file in ct_files])

            tumor_positions = [position for sublist in results for position in sublist]

            self.all_tumor_positions = np.array(tumor_positions)
        except Exception as e:
            print("Error occurred while loading data:", e)

    def load_data_and_fit_gmm(self, data_folder, optimal_components):
        """
        Loads data and fits GMM model.
        """
        if not self.has_fitted_gmm:
            try:
                self.load_data(data_folder)
                self.fit_gmm_model(self.all_tumor_positions, optimal_components)
                self.has_fitted_gmm = True
            except Exception as e:
                print("Error occurred while loading data and fitting GMM model:", e)


    @staticmethod
    def analyze_tumor_location(label, liver_label=1, tumor_label=2):
        """
        Analyzes tumor location from label data.
        """
        try:
            label_data = label.get_fdata()

            liver_mask = np.zeros_like(label_data).astype(np.int16)
            tumor_mask = np.zeros_like(label_data).astype(np.int16)
            liver_mask[label_data == liver_label] = 1
            liver_mask[label_data == tumor_label] = 1
            tumor_mask[label_data == tumor_label] = 1

            tumor_positions = []

            if len(np.unique(tumor_mask)) > 1:
                label_numeric, gt_N = ndimage.label(tumor_mask)
                for segid in range(1, gt_N + 1):
                    extracted_label_numeric = np.uint8(label_numeric == segid)
                    clot_size = np.sum(extracted_label_numeric)
                    if clot_size < 8:
                        continue
                    # center_of_mass = ndimage.measurements.center_of_mass(extracted_label_numeric)
                    # Get coordinates of tumor voxels
                    x, y, z = np.where(extracted_label_numeric)

                    # Initial guess for the parameters of the ellipsoid (center and radii)
                    initial_guess = [np.mean(x), np.mean(y), np.mean(z), 1, 1, 1]

                    # Residual function to minimize
                    def residual(params):
                        cx, cy, cz, a, b, c = params
                        distances = np.sqrt(((x - cx) / a)**2 + ((y - cy) / b)**2 + ((z - cz) / c)**2) - 1
                        return distances

                    # Fit ellipsoid parameters
                    result = least_squares(residual, initial_guess)

                    # Extract center of the ellipsoid (which represents tumor position)
                    tumor_position = result.x[:3]
                    tumor_positions.append(tuple(tumor_position))

            return tumor_positions

        except Exception as e:
            print("Error occurred while analyzing tumor location:", e)
            return []

    def get_gmm_model(self):
        """
        Returns the trained GMM model.
        """
        return self.gmm_model
import os
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from sklearn.mixture import GaussianMixture


class TumorAnalyzer:
    def __init__(self):
        self.all_tumor_positions = None
        self.gmm_model = None
        self.gmm_flag = False
        self.healthy_ct = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]

    def fit_gmm_model(self, train_data, val_data, optimal_components, max_iter=150):
        """
        Fits a Gaussian Mixture Model to the given data.
        """
        try:
            self.gmm_model = GaussianMixture(
                n_components=optimal_components,
                covariance_type='full',
                init_params='kmeans',
                tol=0.00005,
                max_iter=max_iter
            )

            prev_score = float('-inf')
            for iter in range(max_iter):
                self.gmm_model.fit(train_data)
                val_score = self.gmm_model.score(val_data)
                if val_score < prev_score:
                    print("Validation score decreased. Stopping early.")
                    break
                prev_score = val_score
        except Exception as e:
            print("Error occurred while fitting GMM model:", e)

    @staticmethod
    def process_file(ct_file, data_folder, healthy_ct):
        try:
            if ct_file.startswith("._"):
                return []

            img_path = os.path.join(data_folder, "img", ct_file)
            label_path = os.path.join(data_folder, "label", ct_file)

            ct_index = int(ct_file.split('_')[1].split('.')[0])
            if ct_index in healthy_ct:
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
                results = pool.starmap(TumorAnalyzer.process_file, [(ct_file, data_folder, self.healthy_ct) for ct_file in ct_files])

            tumor_positions = [position for sublist in results for position in sublist]

            self.all_tumor_positions = np.array(tumor_positions)
        except Exception as e:
            print("Error occurred while loading data:", e)

    def split_train_val(self, test_size=0.2, random_state=42):
        """
        Splits the tumor positions into training and validation sets.
        """
        train_data, val_data = train_test_split(self.all_tumor_positions, test_size=test_size, random_state=random_state)
        return train_data, val_data

    def gmm_starter(self, data_folder, optimal_components, test_size=0.2, random_state=42):
        """
        Loads data, prepares training and validation sets, and fits GMM model with early stopping.
        """
        if not self.gmm_flag:
            try:
                self.load_data(data_folder)
                train_data, val_data = self.split_train_val(test_size=test_size, random_state=random_state)
                self.fit_gmm_model(train_data, val_data, optimal_components, 150)
                self.gmm_flag = True
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

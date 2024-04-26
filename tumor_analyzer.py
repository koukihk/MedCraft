import os
import glob
from tqdm import tqdm
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import least_squares
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


class TumorAnalyzer:
    def __init__(self):
        self.all_tumor_positions = None
        self.gmm_model = None
        self.gmm_flag = False
        self.healthy_ct = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
        self.healthy_median_size = (253, 215, 162)
        self.unhealthy_median_size = (378, 229, 107)
        self.healthy_mean_size = (287, 242, 154)
        self.unhealthy_mean_size = (282, 244, 143)
        self.target_volume = self.healthy_mean_size

    def fit_gmm_model(self, train_data, val_data, optimal_components, max_iter=500, early_stopping=True, tol=0.00001,
                      patience=5):
        """
        Fits a Gaussian Mixture Model to the given data.
        """
        try:
            self.gmm_model = GaussianMixture(
                n_components=optimal_components,
                covariance_type='full',
                init_params='k-means++',
                tol=tol,
                max_iter=max_iter
            )

            best_score = float('-inf')
            best_params = None
            no_improvement_count = 0

            for iter in range(max_iter):
                self.gmm_model.fit(train_data)
                val_score = self.gmm_model.score(val_data)

                if val_score > best_score:
                    best_score = val_score
                    best_params = self.gmm_model.get_params()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if early_stopping and no_improvement_count >= patience:
                    print("Validation score did not improve for {} iterations. Rolling back to best model.".format(
                        patience))
                    self.gmm_model.set_params(**best_params)
                    break

        except Exception as e:
            print("Error occurred while fitting GMM model:", e)

    @staticmethod
    def voxel2R(A):
        return (np.array(A) / 4 * 3 / np.pi) ** (1 / 3)

    @staticmethod
    def pixel2voxel(A, res=[0.75, 0.75, 0.5]):
        return np.array(A) * (res[0] * res[1] * res[2])

    @staticmethod
    def analyze_tumor_type(data_dir='datafolds/04_LiTS/label/', output_save_dir='datafolds/04_LiTS/'):
        tiny, small, medium, large = 0, 0, 0, 0
        total_clot_size = []
        total_clot_size_mmR = []
        valid_ct_name = []
        label_paths = glob.glob(os.path.join(data_dir, 'liver_*.nii.gz'))
        label_paths.sort()

        result_file = os.path.join(output_save_dir, 'tumor_type_result.txt')
        with open(result_file, 'w') as f:
            for label_path in label_paths:
                print('label_path', label_path)
                file_name = os.path.basename(label_path)

                label = nib.load(label_path)
                pixdim = label.header['pixdim']
                spacing_mm = tuple(pixdim[1:4])
                raw_label = label.get_fdata()

                tumor_mask = np.zeros_like(raw_label).astype(np.int16)
                organ_mask = np.zeros_like(raw_label).astype(np.int16)
                organ_mask[raw_label == 1] = 1
                organ_mask[raw_label == 2] = 1
                tumor_mask[raw_label == 2] = 1

                if len(np.unique(tumor_mask)) > 1:
                    label_numeric, gt_N = ndimage.label(tumor_mask)
                    for segid in range(1, gt_N + 1):
                        extracted_label_numeric = np.uint8(label_numeric == segid)
                        clot_size = np.sum(extracted_label_numeric)
                        if clot_size < 8:
                            continue
                        clot_size_mm = TumorAnalyzer.pixel2voxel(clot_size, spacing_mm)
                        clot_size_mmR = TumorAnalyzer.voxel2R(clot_size_mm)
                        print('tumor clot_size_mmR', clot_size_mmR)

                        if clot_size_mmR <= 10:
                            tiny += 1
                        elif 10 < clot_size_mmR <= 25:
                            small += 1
                        elif 25 < clot_size_mmR <= 50:
                            medium += 1
                        else:
                            large += 1

                        total_clot_size.append(clot_size)
                        total_clot_size_mmR.append(clot_size_mmR)
                        if not file_name in valid_ct_name:
                            valid_ct_name.append(file_name)

                        f.write(f"File Name: {file_name}, "
                                f"Tumor Size (pixel): {clot_size}, "
                                f"Tumor Size (voxel): {clot_size}, "
                                f"Tumor Size (mmR): {clot_size_mmR}\n")

        total = tiny + small + medium + large
        f.write(
            f"Tiny: {tiny} ({tiny / total:.2%}), "f"Small: {small} ({small / total:.2%}), "
            f"Medium: {medium} ({medium / total:.2%}), "f"Large: {large} ({large / total:.2%})")

        return tiny, small, medium, large, total_clot_size, total_clot_size_mmR, valid_ct_name

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
            positions = TumorAnalyzer.analyze_tumor_location(label, (287, 242, 154), 1, 2)

            return positions
        except Exception as e:
            print("Error occurred while processing file", ct_file, ":", e)
            return []

    def load_data(self, data_folder, parallel=False):
        """
        Loads CT scan images and corresponding tumor labels from the specified data folder.
        """
        try:
            ct_files = sorted(os.listdir(os.path.join(data_folder, "img")))

            if parallel:
                with Pool() as pool:
                    results = list(tqdm(pool.imap(TumorAnalyzer.process_file,
                                                  [(ct_file, data_folder, self.healthy_ct) for ct_file in ct_files]),
                                        total=len(ct_files), desc="Loading dataset"))
            else:
                results = []
                for ct_file in tqdm(ct_files, total=len(ct_files), desc="Loading dataset"):
                    result = TumorAnalyzer.process_file(ct_file, data_folder, self.healthy_ct)
                    results.append(result)

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
                self.load_data(data_folder, parallel=False)
                train_data, val_data = self.split_train_val(test_size=test_size, random_state=random_state)
                self.fit_gmm_model(train_data, val_data, optimal_components, 500)
                self.gmm_flag = True
            except Exception as e:
                print("Error occurred while loading data and fitting GMM model:", e)

    @staticmethod
    def crop_mask(mask_scan):
        """
        Crops the volume to get the liver mask.
        """
        # for speed_generate_tumor, we only send the liver part into the generate program
        x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
        y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
        z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

        # shrink the boundary
        x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
        y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
        z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

        liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

        return liver_mask

    @staticmethod
    def resize_mask(volume, new_shape):
        """
        Resizes the volume on given shape.
        """
        x_old, y_old, z_old = volume.shape
        x_new, y_new, z_new = new_shape

        # Create grid for interpolation
        x = np.linspace(0, x_old - 1, x_old)
        y = np.linspace(0, y_old - 1, y_old)
        z = np.linspace(0, z_old - 1, z_old)

        new_x = np.linspace(0, x_old - 1, x_new)
        new_y = np.linspace(0, y_old - 1, y_new)
        new_z = np.linspace(0, z_old - 1, z_new)

        # Create interpolation function
        interpolator = interpolate.RegularGridInterpolator((x, y, z), volume, method='linear', bounds_error=False,
                                                           fill_value=0)

        # Interpolate volume
        new_volume = interpolator((new_x[:, None, None], new_y[None, :, None], new_z[None, None, :]))

        return np.round(new_volume).astype(int)

    @staticmethod
    def fit_ellipsoid_center(extracted_label_numeric):
        """
        Fits an ellipsoid, and gets its center.
        """
        # Residual function to minimize
        def residual(params):
            cx, cy, cz, a, b, c = params
            distances = np.sqrt(((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 + ((z - cz) / c) ** 2) - 1
            return distances

        # Get coordinates of tumor voxels
        x, y, z = np.where(extracted_label_numeric)

        # Initial guess for the parameters of the ellipsoid (center and radii)
        initial_guess = [np.mean(x), np.mean(y), np.mean(z), 1, 1, 1]

        # Fit ellipsoid parameters
        result = least_squares(residual, initial_guess)
        # Extract center of the ellipsoid (which represents tumor position)
        tumor_position = result.x[:3]

        return tumor_position

    @staticmethod
    def analyze_tumor_location(label, target_volume=(287, 242, 154), liver_label=1, tumor_label=2):
        """
        Analyzes tumor location from label data.
        """
        try:
            label_data = label.get_fdata()

            organ_mask = TumorAnalyzer.crop_mask(label_data)
            organ_mask = TumorAnalyzer.resize_mask(organ_mask, target_volume)

            liver_mask = np.zeros_like(organ_mask).astype(np.int16)
            tumor_mask = np.zeros_like(organ_mask).astype(np.int16)
            liver_mask[organ_mask == liver_label] = 1
            liver_mask[organ_mask == tumor_label] = 1
            tumor_mask[organ_mask == tumor_label] = 1

            tumor_positions = []

            if len(np.unique(tumor_mask)) > 1:
                label_numeric, gt_N = ndimage.label(tumor_mask)
                for segid in range(1, gt_N + 1):
                    extracted_label_numeric = np.uint8(label_numeric == segid)
                    clot_size = np.sum(extracted_label_numeric)
                    if clot_size < 8:
                        continue
                    center_of_mass = ndimage.measurements.center_of_mass(extracted_label_numeric)
                    if any(coord < 0 for coord in center_of_mass):
                        continue
                    tumor_positions.append(center_of_mass)

            return tumor_positions

        except Exception as e:
            print("Error occurred while analyzing tumor location:", e)
            return []

    def get_gmm_model(self):
        """
        Returns the trained GMM model.
        """
        return self.gmm_model

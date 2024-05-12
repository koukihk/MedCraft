import glob
import os
import pickle
import random
import string
import warnings
from multiprocessing import Pool, cpu_count

import nibabel as nib
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Tumor:
    def __init__(self, position=None, type=None, filename=None):
        self.position = position  # relative position
        self.type = type  # one of ['tiny', 'small', 'medium', 'large']
        self.filename = filename  # liver_*.nii.gz

    def __repr__(self):
        return f"Tumor(position={self.position}, type={self.type}, filename={self.filename})"


class TumorAnalyzer:
    def __init__(self):
        self.all_tumors = None
        self.gmm_model = None
        self.gmm_model_global = None
        self.gmm_model_tiny = None
        self.gmm_model_non_tiny = None
        self.gmm_flag = False
        self.healthy_ct = [32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]
        self.healthy_median_size = (253, 215, 162)
        self.unhealthy_median_size = (378, 229, 107)
        self.healthy_mean_size = (287, 242, 154)
        self.unhealthy_mean_size = (282, 244, 143)
        self.target_spacing = (0.86950004, 0.86950004, 0.923077)
        self.target_volume = self.healthy_mean_size

    def fit_gmm_model(self, train_tumors, val_tumors, optimal_components, cov_type='tied', tol=0.00001, max_iter=500,
                      early_stopping=True, patience=3):
        """
        Fits a Gaussian Mixture Model to the given data.
        """
        train_position = np.array([tumor.position for tumor in train_tumors])
        val_position = np.array([tumor.position for tumor in val_tumors])

        self.gmm_model = GaussianMixture(
            n_components=optimal_components,
            covariance_type=cov_type,
            init_params='k-means++',
            tol=tol,
            max_iter=max_iter
        )

        if early_stopping:
            best_score = float('-inf')
            best_params = None
            no_improvement_count = 0

            for iter in range(max_iter):
                self.gmm_model.fit(train_position)
                val_score = self.gmm_model.score(val_position)

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
        else:
            train_position = np.concatenate((train_position, val_position))
            self.gmm_model.fit(train_position)

    @staticmethod
    def process_file(ct_file, data_folder):
        img_path = os.path.join(data_folder, "img", ct_file)
        label_path = os.path.join(data_folder, "label", ct_file)

        if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
            return [], []

        tumors = TumorAnalyzer.analyze_tumors(label_path, (287, 242, 154), 2, False)
        return tumors

    def load_data(self, data_folder, parallel=False):
        """
        Loads CT scan images and corresponding tumor labels from the specified data folder.
        """
        ct_files = sorted(os.listdir(os.path.join(data_folder, "label")))
        expected_count = len(ct_files) // 2 - len(self.healthy_ct)
        ct_files = [ct_file for ct_file in ct_files
                    if not ct_file.startswith("._")
                    and int(ct_file.split('_')[1].split('.')[0]) not in self.healthy_ct]

        if len(ct_files) != expected_count:
            warnings.warn(f"Expected {expected_count} files after filtering, but found {len(ct_files)}.",
                          Warning)

        all_tumors = []
        if parallel:
            max_processes = min(cpu_count(), 6)
            with Pool(max_processes) as pool:
                results = []
                for ct_file in ct_files:
                    results.append(pool.apply_async(TumorAnalyzer.process_file, (ct_file, data_folder)))

                for result in tqdm(results, total=len(results), desc="Processing dataset"):
                    tumors = result.get()
                    all_tumors.extend(tumors)

        else:
            for ct_file in tqdm(ct_files, total=len(ct_files), desc="Processing dataset"):
                tumors = TumorAnalyzer.process_file(ct_file, data_folder)
                all_tumors.extend(tumors)

        self.all_tumors = all_tumors

        tumor_count = len(all_tumors)
        type_count = {'tiny': 0, 'small': 0, 'medium': 0, 'large': 0}

        for tumor in all_tumors:
            type_count[tumor.type] += 1
        type_proportions = {tumor_type: count / tumor_count for tumor_type, count in type_count.items()}

        print("Total number of tumors:", tumor_count)
        print("Tumor counts by type: " + ", ".join(
            [f"{tumor_type}: {count}" for tumor_type, count in type_count.items()]))
        print("Tumor type proportions:",
              ", ".join([f"{tumor_type}: {proportion:.2%}" for tumor_type, proportion in type_proportions.items()]))

    def split_train_val(self, test_size=0.2, random_state=42):
        """
        Splits the tumor positions into training and validation sets.
        """
        tumors = self.all_tumors
        train_tumors, val_tumors = train_test_split(tumors, test_size=test_size, random_state=random_state)

        return train_tumors, val_tumors

    def gmm_starter(self, data_folder, optimal_components, split=False, early_stopping=True, parallel=False):
        """
        Loads data, prepares training and validation sets, and fits GMM model with early stopping.
        """

        def generate_random_str(length=6):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

        test_size = 0.2
        random_state = 42
        cov_type = 'diag'
        tol = 0.00001
        max_iter = 500
        patience = 3

        if not self.gmm_flag:
            os.makedirs(f'gmm/{cov_type}', exist_ok=True)
            print_mode = "global" if not split else "split"
            print(f'use {print_mode} mode: {optimal_components}')
            self.load_data(data_folder, parallel=parallel)

            if split:
                all_tiny_tumors = [tumor for tumor in self.all_tumors if tumor.type == 'tiny']
                all_non_tiny_tumors = [tumor for tumor in self.all_tumors if tumor.type != 'tiny']
                train_tiny_tumors, val_tiny_tumors = train_test_split(all_tiny_tumors, test_size=test_size,
                                                                      random_state=random_state)
                train_non_tiny_tumors, val_non_tiny_tumors = train_test_split(all_non_tiny_tumors, test_size=test_size,
                                                                              random_state=random_state)

                nc_tiny, nc_non_tiny = optimal_components

                for tumor_type, train_tumors, val_tumors, nc in [("tiny", train_tiny_tumors, val_tiny_tumors, nc_tiny),
                                                                 (
                                                                 "non_tiny", train_non_tiny_tumors, val_non_tiny_tumors,
                                                                 nc_non_tiny)]:
                    self.fit_gmm_model(train_tumors, val_tumors, nc, cov_type, tol, max_iter, early_stopping, patience)
                    gmm_model_name = f'gmm_model_{tumor_type}_{nc}_{generate_random_str()}.pkl'
                    with open(os.path.join('gmm', cov_type, gmm_model_name), 'wb') as f:
                        pickle.dump(self.gmm_model, f)
                    if tumor_type == "tiny":
                        self.gmm_model_tiny = self.gmm_model
                    else:
                        self.gmm_model_non_tiny = self.gmm_model
                    print(f"{tumor_type.capitalize()} GMM saved successfully: gmm/{cov_type}/{gmm_model_name}")

            else:
                train_tumors, val_tumors = self.split_train_val(test_size=test_size, random_state=random_state)
                nc = optimal_components[0]
                self.fit_gmm_model(train_tumors, val_tumors, nc, cov_type, tol, max_iter, early_stopping, patience)
                gmm_model_name = f'gmm_model_global_{nc}_{generate_random_str()}.pkl'
                with open(os.path.join('gmm', cov_type, gmm_model_name), 'wb') as f:
                    pickle.dump(self.gmm_model, f)
                    self.gmm_model_global = self.gmm_model
                print(f"Global GMM saved successfully: gmm/{cov_type}/{gmm_model_name}")

            self.gmm_flag = True

    @staticmethod
    def analyze_tumors_shape(data_dir='datafolds/04_LiTS/label/', output_save_dir='datafolds/04_LiTS/',
                             file_reg='liver_*.nii.gz'):
        label_paths = glob.glob(os.path.join(data_dir, file_reg))
        label_paths.sort()

        valid_ct_name = []

        result_file = os.path.join(output_save_dir, 'tumor_shape_result.txt')
        with open(result_file, 'w') as f:
            for label_path in label_paths:
                print('label_path', label_path)
                file_name = os.path.basename(label_path)

                label = nib.load(label_path)
                raw_label = label.get_fdata()

                tumor_mask = np.zeros_like(raw_label).astype(np.int16)
                tumor_mask[raw_label == 2] = 1

                if len(np.unique(tumor_mask)) > 1:
                    label_numeric, gt_N = ndimage.label(tumor_mask)
                    for segid in range(1, gt_N + 1):
                        extracted_label_numeric = np.uint8(label_numeric == segid)
                        clot_size = np.sum(extracted_label_numeric)
                        if clot_size < 8:
                            continue
                        coords = np.array(np.where(extracted_label_numeric)).T
                        centroid = np.mean(coords, axis=0)
                        distances = cdist([centroid], coords)
                        x_radius = np.max(distances[:, 0])
                        y_radius = np.max(distances[:, 1])
                        z_radius = np.max(distances[:, 2])

                        print('Tumor Shape - X radius:', x_radius, 'Y radius:', y_radius, 'Z radius:', z_radius)
                        f.write(f"Tumor Shape - X radius: {x_radius}, Y radius: {y_radius}, Z radius: {z_radius}\n")

                    if not file_name in valid_ct_name:
                        valid_ct_name.append(file_name)

        with open(result_file, 'a') as f:
            f.write(f"Valid_ct: {len(valid_ct_name)}\n")

    @staticmethod
    def analyze_tumor_type_helper(clot_size, spacing_mm):
        def voxel2R(A):
            return (np.array(A) / 4 * 3 / np.pi) ** (1 / 3)

        def pixel2voxel(A, res=[0.75, 0.75, 0.5]):
            return np.array(A) * (res[0] * res[1] * res[2])

        clot_size_mm = pixel2voxel(clot_size, spacing_mm)
        clot_size_mmR = voxel2R(clot_size_mm)

        if clot_size_mmR <= 10:
            tumor_type = 'tiny'
        elif 10 < clot_size_mmR <= 25:
            tumor_type = 'small'
        elif 25 < clot_size_mmR <= 50:
            tumor_type = 'medium'
        else:
            tumor_type = 'large'
        return tumor_type, clot_size_mm, clot_size_mmR

    @staticmethod
    def analyze_tumors_type(data_dir='datafolds/04_LiTS/label/', output_save_dir='datafolds/04_LiTS/',
                            file_reg='liver_*.nii.gz'):
        tumor_counts = {'tiny': 0, 'small': 0, 'medium': 0, 'large': 0}
        total_clot_size = []
        total_clot_size_mmR = []
        valid_ct_name = []
        label_paths = glob.glob(os.path.join(data_dir, file_reg))
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
                tumor_mask[raw_label == 2] = 1

                if len(np.unique(tumor_mask)) > 1:
                    label_numeric, gt_N = ndimage.label(tumor_mask)
                    for segid in range(1, gt_N + 1):
                        extracted_label_numeric = np.uint8(label_numeric == segid)
                        clot_size = np.sum(extracted_label_numeric)
                        if clot_size < 8:
                            continue
                        tumor_type, clot_size_mm, clot_size_mmR = TumorAnalyzer.analyze_tumor_type_helper(clot_size,
                                                                                                          spacing_mm)
                        print('tumor_clot_size_mmR', clot_size_mmR, 'tumor_type', tumor_type)

                        if tumor_type in tumor_counts:
                            tumor_counts[tumor_type] += 1
                        else:
                            tumor_counts['large'] += 1

                        total_clot_size.append(clot_size)
                        total_clot_size_mmR.append(clot_size_mmR)
                        if not file_name in valid_ct_name:
                            valid_ct_name.append(file_name)

                        f.write(f"File Name: {file_name}, "
                                f"Tumor Size (pixel): {clot_size}, "
                                f"Tumor Size (voxel): {clot_size}, "
                                f"Tumor Size (mmR): {clot_size_mmR},"
                                f"Tumor Type: {tumor_type}\n")

        with open(result_file, 'a') as f:
            f.write(f"Valid_ct: {len(valid_ct_name)}\n")

            total = sum(tumor_counts.values())
            for tumor_type, count in tumor_counts.items():
                f.write(f"{tumor_type.capitalize()}: {count} ({count / total:.2%}), ")

        return tumor_counts['tiny'], tumor_counts['small'], tumor_counts['medium'], tumor_counts[
            'large'], total_clot_size, total_clot_size_mmR, valid_ct_name

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
    def resize_mask(mask_scan, new_shape, target_spacing=(0.86950004, 0.86950004, 0.923077)):
        """
        Resizes the volume on given shape with target spacing.
        """
        x_old, y_old, z_old = mask_scan.shape
        x_new, y_new, z_new = new_shape

        # Compute current spacing
        spacing_x = 1.0
        spacing_y = 1.0
        spacing_z = 1.0

        if x_old > 1:
            spacing_x = (x_old - 1) / (x_old - 1)
        if y_old > 1:
            spacing_y = (y_old - 1) / (y_old - 1)
        if z_old > 1:
            spacing_z = (z_old - 1) / (z_old - 1)

        # Compute scaling factors for new spacing
        scale_x = spacing_x / target_spacing[0]
        scale_y = spacing_y / target_spacing[1]
        scale_z = spacing_z / target_spacing[2]

        # Create grid for interpolation
        x = np.linspace(0, x_old - 1, x_old)
        y = np.linspace(0, y_old - 1, y_old)
        z = np.linspace(0, z_old - 1, z_old)

        new_x = np.linspace(0, x_old - 1, int(x_new * scale_x))
        new_y = np.linspace(0, y_old - 1, int(y_new * scale_y))
        new_z = np.linspace(0, z_old - 1, int(z_new * scale_z))

        # Create interpolation function
        interpolator = interpolate.RegularGridInterpolator((x, y, z), mask_scan, method='nearest', bounds_error=False,
                                                           fill_value=0)

        # Interpolate volume
        new_volume = interpolator((new_x[:, None, None], new_y[None, :, None], new_z[None, None, :]))

        return np.round(new_volume).astype(int)

    @staticmethod
    def analyze_tumors(label_path, target_volume=(287, 242, 154), tumor_label=2, mapper=False):
        """
        Analyzes tumor information from label data.
        """

        def tumor_mapper(extracted_tumor):
            return extracted_tumor

        file_name = os.path.basename(label_path)
        label = nib.load(label_path)
        pixdim = label.header['pixdim']
        spacing_mm = tuple(pixdim[1:4])
        label_data = label.get_fdata()

        organ_mask = TumorAnalyzer.crop_mask(label_data)
        organ_mask = TumorAnalyzer.resize_mask(organ_mask, target_volume)

        tumor_mask = np.zeros_like(organ_mask).astype(np.int16)
        tumor_mask[organ_mask == tumor_label] = 1

        tumors = []

        if len(np.unique(tumor_mask)) > 1:
            label_numeric, gt_N = ndimage.label(tumor_mask)
            for segid in range(1, gt_N + 1):
                extracted_label_numeric = np.uint8(label_numeric == segid)
                if mapper:
                    mapped_label_binary = tumor_mapper(extracted_label_numeric)
                    clot_size = np.sum(mapped_label_binary)
                else:
                    clot_size = np.sum(extracted_label_numeric)
                if clot_size < 8:
                    continue
                tumor_position = ndimage.measurements.center_of_mass(extracted_label_numeric)
                # if any(coord < 0 for coord in tumor_position):
                #     continue
                tumor_type, _, _ = TumorAnalyzer.analyze_tumor_type_helper(clot_size, spacing_mm)
                tumor = Tumor(position=tumor_position, type=tumor_type, filename=file_name)
                tumors.append(tumor)

        return tumors

    def get_gmm_model(self, model_type='global'):
        """
        Returns the trained GMM model.
        """
        models = {
            'tiny': self.gmm_model_tiny,
            'non_tiny': self.gmm_model_non_tiny,
            'global': self.gmm_model_global
        }

        return models.get(model_type)



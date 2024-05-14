import glob
import os
import pickle
import random
import string
import warnings
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
from scipy import interpolate
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class GMMPlotter:
    @staticmethod
    def gmm2plt(gmm_model, num_samples=500):
        """
        Plot a 3D visualization of a Gaussian Mixture Model (GMM).

        Parameters:
        - gmm_model: A fitted Gaussian Mixture Model with attributes means_, covariances_, weights_, and covariance_type.
        - num_samples: The total number of samples to generate from the GMM.
        """

        def plot_gmm_component(ax, mean, covariance, color, num_points=50):
            """
            Plot the ellipsoid representing the GMM component defined by mean and covariance.

            Parameters:
            - ax: The 3D axis to plot on.
            - mean: Mean of the Gaussian component.
            - covariance: Covariance matrix of the Gaussian component.
            - color: Color of the ellipsoid.
            - num_points: Number of points to use for plotting the ellipsoid.
            """
            u = np.linspace(0, 2 * np.pi, num_points)
            v = np.linspace(0, np.pi, num_points)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(num_points), np.cos(v))

            xyz = np.dot(np.vstack((x.flatten(), y.flatten(), z.flatten())).T, np.linalg.cholesky(covariance).T) + mean

            x = xyz[:, 0].reshape(num_points, num_points)
            y = xyz[:, 1].reshape(num_points, num_points)
            z = xyz[:, 2].reshape(num_points, num_points)

            ax.plot_surface(x, y, z, color=color, alpha=0.3)

        def get_covariance_matrix(covariances, index, covariance_type):
            """
            Retrieve the covariance matrix for the specified component.

            Parameters:
            - covariances: Covariance matrices from the GMM.
            - index: Index of the component.
            - covariance_type: Type of the covariance matrix (full, tied, diag, spherical).

            Returns:
            - The covariance matrix for the specified component.
            """
            if covariance_type == 'full':
                return covariances[index]
            elif covariance_type == 'tied':
                return covariances
            elif covariance_type == 'diag':
                return np.diag(covariances[index])
            elif covariance_type == 'spherical':
                return np.eye(len(covariances[index])) * covariances[index]

        # Set up the plot
        xlim = (0, 350)
        ylim = (0, 600)
        zlim = (0, 200)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        means = gmm_model.means_
        covariances = gmm_model.covariances_
        weights = gmm_model.weights_
        covariance_type = gmm_model.covariance_type

        for i, weight in enumerate(weights):
            cov_matrix = get_covariance_matrix(covariances, i, covariance_type)

            # Generate random samples according to the component's parameters
            samples = np.random.multivariate_normal(means[i], cov_matrix, size=max(1, int(num_samples * weight)))

            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=10, alpha=0.4, label=f'Component {i + 1}',
                       color=f'C{i % 10}')

            plot_gmm_component(ax, means[i], cov_matrix, color=f'C{i % 10}')

        # Customize axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        ax.legend(loc='upper right')

        # Information text
        info_text = '\n'.join(
            [f'Component {i + 1}:\nMean: {np.round(means[i], 2)}\nCovariance: {np.round(covariances[i], 2)}' for i in
             range(len(weights))])
        ax.text2D(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()


class Tumor:
    def __init__(self, position=None, type=None, filename=None):
        self.position = position  # relative position
        self.type = type  # one of ['tiny', 'small', 'medium', 'large']

    def __repr__(self):
        return f"Tumor(position={self.position}, type={self.type})"


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
        train_positions = np.array([tumor.position for tumor in train_tumors])
        val_positions = np.array([tumor.position for tumor in val_tumors])

        # debug
        n_components_range = range(1, 6)
        aic_scores = []
        bic_scores = []

        for n_components in n_components_range:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                init_params='k-means++',
                tol=tol,
                max_iter=max_iter
            )
            debug_positions = np.concatenate((train_positions, val_positions))
            gmm.fit(debug_positions)
            aic = gmm.aic(debug_positions)
            bic = gmm.bic(debug_positions)
            aic_scores.append(aic)
            bic_scores.append(bic)

            print(f"Number of components: {n_components}, AIC: {aic}, BIC: {bic}")

        best_aic_idx = np.argmin(aic_scores)
        best_bic_idx = np.argmin(bic_scores)
        best_aic = n_components_range[best_aic_idx]
        best_bic = n_components_range[best_bic_idx]

        print("Best number of components based on AIC:", best_aic)
        print("Best number of components based on BIC:", best_bic)

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
                self.gmm_model.fit(train_positions)
                val_score = self.gmm_model.score(val_positions)

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
            train_positions = np.concatenate((train_positions, val_positions))
            self.gmm_model.fit(train_positions)

    @staticmethod
    def process_file(ct_file, data_folder):
        img_path = os.path.join(data_folder, "img", ct_file)
        label_path = os.path.join(data_folder, "label", ct_file)

        if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
            return [], []

        tumors = TumorAnalyzer.analyze_tumors(label_path, (287, 242, 154), 2)
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
        interpolator = interpolate.RegularGridInterpolator((x, y, z), volume, method='nearest', bounds_error=False,
                                                           fill_value=0)

        # Interpolate volume
        new_volume = interpolator((new_x[:, None, None], new_y[None, :, None], new_z[None, None, :]))

        return np.round(new_volume).astype(int)

    @staticmethod
    def resize_mask_spacing(mask_scan, new_shape, origin_spacing, target_spacing=(0.86950004, 0.86950004, 0.923077)):
        """
        Resizes the volume to the given shape with target spacing.
        """
        old_shape = mask_scan.shape
        old_spacing = origin_spacing

        scale = [old_spacing[i] / target_spacing[i] for i in range(3)]

        new_shape_scaled = [int(old_shape[i] * scale[i]) for i in range(3)]

        new_volume = ndimage.zoom(mask_scan, zoom=scale, mode='nearest', order=0)

        pad_width = [(0, max(0, new_shape[i] - new_shape_scaled[i])) for i in range(3)]
        new_volume = np.pad(new_volume, pad_width, mode='constant')

        new_volume = new_volume[:new_shape[0], :new_shape[1], :new_shape[2]]

        new_volume[new_volume > 1] = 1

        return new_volume.astype(int)

    @staticmethod
    def analyze_tumors(label_path, target_volume=(287, 242, 154), tumor_label=2):
        """
        Analyzes tumor information from label data.
        """
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
                clot_size = np.sum(extracted_label_numeric)
                if clot_size < 8:
                    continue
                tumor_position = ndimage.measurements.center_of_mass(extracted_label_numeric)
                tumor_type, _, _ = TumorAnalyzer.analyze_tumor_type_helper(clot_size, spacing_mm)
                tumor = Tumor(position=tumor_position, type=tumor_type)
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


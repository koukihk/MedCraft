### Tumor Generateion
import random

import cv2
import elasticdeform
import numpy as np
import pywt
from noise import snoise3
from scipy.ndimage import gaussian_filter, median_filter, sobel
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from skimage.filters import gabor


def generate_complex_noise(mask_shape, scale=10):
    a = np.zeros(mask_shape)
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            for k in range(mask_shape[2]):
                a[i, j, k] = snoise3(i / scale, j / scale, k / scale)

    # Normalize to 0-1 range
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    return a


def enhance_contrast(image):
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    enhanced_image = exposure.equalize_adapthist(image, clip_limit=0.03)
    return enhanced_image


def apply_gabor_filter(image):
    # Apply Gabor filter with different frequencies and angles
    filtered_real, filtered_imag = gabor(image, frequency=0.6)

    # Combine the real and imaginary parts to get the final filtered image
    gabor_filtered = np.sqrt(filtered_real ** 2 + filtered_imag ** 2)

    return gabor_filtered

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob, median_filter_size=3):
    noisy_image = np.copy(image.astype(float))

    salt_value = noisy_image.max()
    pepper_value = noisy_image.min()

    num_salt = np.ceil(salt_prob * noisy_image.size)
    num_pepper = np.ceil(pepper_prob * noisy_image.size)

    # Generate salt noise coordinates
    salt_coords = [np.random.randint(0, dim, int(num_salt)) for dim in noisy_image.shape]
    noisy_image[tuple(salt_coords)] = salt_value

    # Generate pepper noise coordinates
    pepper_coords = [np.random.randint(0, dim, int(num_pepper)) for dim in noisy_image.shape]
    noisy_image[tuple(pepper_coords)] = pepper_value

    # Apply median filter to the noisy image
    noisy_image = median_filter(noisy_image, size=median_filter_size)

    return noisy_image


def generate_prob_function(mask_shape):
    sigma = np.random.uniform(3, 15)
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))

    # Gaussian filter
    # this taks some time
    a_2 = gaussian_filter(a, sigma=sigma)

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    return a


# first generate 5*200*200*200
def get_texture(mask_shape):
    # get the prob function
    a = generate_prob_function(mask_shape)

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))

    # if a(x) > random_sample(x), set b(x) = 1
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter

    # Gaussian filter
    if np.random.uniform() < 0.7:
        sigma_b = np.random.uniform(3, 5)
    else:
        sigma_b = np.random.uniform(5, 8)

    # this takes some time
    b2 = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12  # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b2, 0, 1)  # 目前是0-1区间

    return Bj


# here we want to get predefined texutre:
def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12  # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)  # 目前是0-1区间

    return Bj

def get_predefined_texture_O(mask_shape, sigma_a, sigma_b):
    simplex_scale = int(np.random.uniform(4, 8))
    a = generate_complex_noise(mask_shape, simplex_scale)
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12  # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)  # 目前是0-1区间

    return Bj

def get_predefined_texture_A(mask_shape, sigma_a, sigma_b):
    # Step 1: Uniform noise generation with larger range for higher contrast
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))

    # Step 2: Apply Gaussian filter to smoothen the noise
    a_2 = gaussian_filter(a, sigma=sigma_a)

    # Step 3: Scale and normalize the filtered noise
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # Step 4: Add localized random noise for heterogeneity
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    b = (a > random_sample).astype(float)

    # Step 5: Apply Gaussian filter to smooth the binary noise
    b = gaussian_filter(b, sigma_b)

    # Step 6: Scaling and clipping for realistic intensity adjustment
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)

    # Step 7: Local texture enhancement (using wavelet)
    # Apply wavelet transform to add finer texture details in different frequency bands
    coeffs = pywt.wavedec2(Bj, wavelet='db4', level=2)
    coeffs[1] = tuple(0.4 * v for v in coeffs[1])  # Retain some high-frequency details
    coeffs[2:] = [tuple(np.zeros_like(v) for v in coeff) for coeff in coeffs[2:]]  # Discard very high frequency noise
    Bj_wavelet = pywt.waverec2(coeffs, wavelet='db4')

    # Normalize back to 0-1 range
    Bj_wavelet = (Bj_wavelet - np.min(Bj_wavelet)) / (np.max(Bj_wavelet) - np.min(Bj_wavelet))

    # Step 8: Optional contrast enhancement (adaptive)
    contrast_adjusted_Bj = np.clip(Bj_wavelet * np.random.uniform(1.1, 1.3), 0, 1)

    return contrast_adjusted_Bj


# here we want to get predefined texutre:
def get_predefined_texture_B(mask_shape, sigma_a, sigma_b):
    # Step 1: Uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    # a = generate_simplex_noise(mask_shape, 0.5)

    # Step 2: Nonlinear diffusion filtering
    a_denoised = denoise_tv_chambolle(a, weight=0.1, multichannel=False)

    # Step 3: Wavelet transform
    coeffs = pywt.wavedec2(a_denoised, wavelet='db4', level=2)
    coeffs[1] = tuple(0.3 * v for v in coeffs[1])  # 保留一些高频细节
    coeffs[2:] = [tuple(np.zeros_like(v) for v in coeff) for coeff in coeffs[2:]]
    a_wavelet_denoised = pywt.waverec2(coeffs, wavelet='db4')

    # Normalize to 0-1
    a_wavelet_denoised = (a_wavelet_denoised - np.min(a_wavelet_denoised)) / (
            np.max(a_wavelet_denoised) - np.min(a_wavelet_denoised))

    # Step 4: Gaussian filter
    # a_2 = gaussian_filter(a, sigma=sigma_a)
    a_2 = gaussian_filter(a_wavelet_denoised, sigma=sigma_a)

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12  # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)  # 目前是0-1区间

    return Bj

def get_predefined_texture_C(mask_shape, sigma_a, sigma_b):
    # Step 1: Complex noise generation (e.g., Simplex noise)
    simplex_scale = int(np.random.uniform(4, 8))
    a = generate_complex_noise(mask_shape, simplex_scale)

    # Step 2: Nonlinear diffusion filtering
    a_denoised = denoise_tv_chambolle(a, weight=0.1, multichannel=False)

    # Step 3: Wavelet transform
    coeffs = pywt.wavedec2(a_denoised, wavelet='db4', level=2)
    coeffs[1] = tuple(0.3 * v for v in coeffs[1])  # 保留一些高频细节
    coeffs[2:] = [tuple(np.zeros_like(v) for v in coeff) for coeff in coeffs[2:]]
    a_wavelet_denoised = pywt.waverec2(coeffs, wavelet='db4')

    # Normalize to 0-1
    a_wavelet_denoised = (a_wavelet_denoised - np.min(a_wavelet_denoised)) / (
            np.max(a_wavelet_denoised) - np.min(a_wavelet_denoised))

    # Step 4: Gaussian filter
    # a_2 = gaussian_filter(a, sigma=sigma_a)
    a_2 = gaussian_filter(a_wavelet_denoised, sigma=sigma_a)

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12  # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)  # 目前是0-1区间

    return Bj

def get_predefined_texture_D(mask_shape, sigma_a, sigma_b):
    # Step 1: Complex noise generation (e.g., Simplex noise)
    simplex_scale = int(np.random.uniform(2, 6))
    a = generate_complex_noise(mask_shape, simplex_scale)

    # Step 2: Nonlinear diffusion filtering with adaptive contrast enhancement
    a_denoised = denoise_tv_chambolle(a, weight=0.1, multichannel=False)
    a_enhanced = enhance_contrast(a_denoised)

    # Step 3: Multi-level wavelet transform with high-frequency detail preservation
    coeffs = pywt.wavedec2(a_enhanced, wavelet='db4', level=3)
    coeffs[1] = tuple(0.5 * v for v in coeffs[1])  # Retain more high-frequency details
    coeffs[2:] = [tuple(np.zeros_like(v) for v in coeff) for coeff in coeffs[2:]]
    a_wavelet_denoised = pywt.waverec2(coeffs, wavelet='db4')

    # Step 4: Apply Gabor filtering or directional filtering for anisotropy
    a_gabor = apply_gabor_filter(a_wavelet_denoised)

    # Normalize to 0-1
    a_normalized = (a_gabor - np.min(a_gabor)) / (np.max(a_gabor) - np.min(a_gabor))

    # Step 5: Gaussian filter
    a_2 = gaussian_filter(a_normalized, sigma=sigma_a)

    # Introduce more randomness and anisotropy
    random_sample = np.random.uniform(0, 1, size=mask_shape)
    b = (a_2 > random_sample).astype(float)
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta * b, 0, 1)

    return Bj

# Step 1: Random select (numbers) location for tumor.
def random_select(mask_scan):
    # we first find z index and then sample point with z slice
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # we need to strict number z's position (0.3 - 0.7 in the middle of liver)
    z = round(random.uniform(0.3, 0.7) * (z_end - z_start)) + z_start

    liver_mask = mask_scan[..., z]

    # erode the mask (we don't want the edge points)
    kernel = np.ones((5, 5), dtype=np.uint8)
    liver_mask = cv2.erode(liver_mask, kernel, iterations=1)

    coordinates = np.argwhere(liver_mask == 1)
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist()  # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points


def get_absolute_coordinate(relative_coordinate, original_shape, target_volume, start):
    x_ratio = original_shape[0] / target_volume[0]
    y_ratio = original_shape[1] / target_volume[1]
    z_ratio = original_shape[2] / target_volume[2]

    absolute_x = relative_coordinate[0] * x_ratio
    absolute_y = relative_coordinate[1] * y_ratio
    absolute_z = relative_coordinate[2] * z_ratio

    absolute_x += start[0]
    absolute_y += start[1]
    absolute_z += start[2]

    return np.array([absolute_x, absolute_y, absolute_z], dtype=float)


def gmm_select(mask_scan, gmm_model=None, max_attempts=600, edge_op="volume"):
    if gmm_model is None:
        potential_point = random_select(mask_scan)
        return potential_point
    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    target_volume = (282, 244, 143)
    start = (x_start, y_start, z_start)

    loop_count = 0
    while loop_count < max_attempts:
        potential_point = gmm_model.sample(1)[0][0]
        if any(coord < 0 for coord in potential_point):
            loop_count += 1
            continue
        potential_point = get_absolute_coordinate(potential_point, liver_mask.shape, target_volume, start)
        potential_point = np.clip(potential_point, 0, np.array(mask_scan.shape) - 1).astype(int)

        if mask_scan[tuple(potential_point)] == 1:
            # Check if the point is not at the edge
            if not is_edge_point(mask_scan, potential_point, edge_op):
                return potential_point

        loop_count += 1

    potential_point = random_select(mask_scan)
    return potential_point


def ellipsoid_select(mask_scan, ellipsoid_model=None, max_attempts=600, edge_op="both"):
    def is_within_middle_z_range(point, z_start, z_end):
        z_length = z_end - z_start
        lower_bound = z_start + 0.3 * z_length
        upper_bound = z_start + 0.7 * z_length
        return lower_bound <= point[2] <= upper_bound

    if ellipsoid_model is None:
        potential_point = random_select(mask_scan)
        return potential_point

    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    target_volume = (282, 244, 143)
    start = (x_start, y_start, z_start)

    loop_count = 0
    while loop_count < max_attempts:
        potential_point = ellipsoid_model.get_random_point()
        if any(coord < 0 for coord in potential_point):
            loop_count += 1
            continue
        potential_point = get_absolute_coordinate(potential_point, liver_mask.shape, target_volume, start)
        potential_point = np.clip(potential_point, 0, np.array(mask_scan.shape) - 1).astype(int)

        if mask_scan[tuple(potential_point)] == 1:
            # Check if the point is not at the edge and within the middle z range
            if not is_edge_point(mask_scan, potential_point, edge_op):
                # and is_within_middle_z_range(potential_point, z_start, z_end)
                return potential_point

        loop_count += 1

    potential_point = random_select(mask_scan)
    return potential_point


def is_edge_point(mask_scan, potential_point, edge_op="both", neighborhood_size=(3, 3, 3), volume_threshold=5,
                  sobel_threshold=405):
    # 定义体积检测方法
    def check_volume():
        # Define the boundaries of the neighborhood around the potential point
        min_bounds = np.maximum(potential_point - np.array(neighborhood_size) // 2, 0)
        max_bounds = np.minimum(potential_point + np.array(neighborhood_size) // 2, np.array(mask_scan.shape) - 1)

        # Extract the neighborhood volume from the mask scan
        neighborhood_volume = mask_scan[min_bounds[0]:max_bounds[0] + 1,
                                        min_bounds[1]:max_bounds[1] + 1,
                                        min_bounds[2]:max_bounds[2] + 1]

        # Count the number of liver voxels in the neighborhood
        liver_voxel_count = np.sum(neighborhood_volume == 1)

        # Check if the liver voxel count is below the threshold
        return liver_voxel_count < volume_threshold

    # 定义Sobel检测方法
    def check_sobel():
        # Apply Sobel filter to detect edges
        sobel_x = sobel(mask_scan, axis=0)
        sobel_y = sobel(mask_scan, axis=1)
        sobel_z = sobel(mask_scan, axis=2)

        # Calculate the magnitude of the gradient
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2 + sobel_z ** 2)

        # Determine if the potential point is near an edge
        gradient_value = gradient_magnitude[tuple(potential_point)]

        return gradient_value > sobel_threshold

    # 根据选择的操作模式判断是否为边缘点
    if edge_op == "volume":
        return check_volume()
    elif edge_op == "sobel":
        return check_sobel()
    elif edge_op == "any":
        return check_volume() and check_sobel()
    elif edge_op == "both":
        return check_volume() or check_sobel()
    elif edge_op == "none":
        return False
    else:
        raise ValueError("Invalid edge_op option. Choose from 'volume', 'sobel', 'any', 'both' or 'none'.")


def get_sphere(r):
    """
    r is the radius of the sphere.
    Returns a 3D numpy array representing the sphere.
    """
    # Create a volume that's large enough to contain the sphere
    sh = (4 * r, 4 * r, 4 * r)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)

    # Center point of the volume
    com = np.array([2 * r, 2 * r, 2 * r])

    # Calculate the bounding box
    bboxl = np.floor(com - r).clip(0, None).astype(int)
    bboxh = (np.ceil(com + r) + 1).clip(None, sh).astype(int)

    # Extract the region of interest (ROI)
    roi = out[tuple(map(slice, bboxl, bboxh))]
    roiaux = aux[tuple(map(slice, bboxl, bboxh))]

    # Create a normalized grid
    x, y, z = np.ogrid[tuple(map(slice, (bboxl - com) / r, (bboxh - com - 1) / r, 1j * (bboxh - bboxl)))]

    # Calculate the distance from each point to the center
    dst = (x ** 2 + y ** 2 + z ** 2).clip(0, None)

    # Create a mask for points inside the sphere
    mask = dst <= 1

    # Set points inside the sphere to 1
    roi[mask] = 1

    # Update the auxiliary array (though not strictly necessary for a sphere)
    np.copyto(roiaux, 1 - dst, where=mask)

    return out


# Step 2 : generate the ellipsoid
def get_ellipsoid(x, y, z, body="ellipsoid"):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    sh = (4 * x, 4 * y, 4 * z)
    out = np.zeros(sh, int)
    aux = np.zeros(sh)
    radii = np.array([x, y, z])
    com = np.array([2 * x, 2 * y, 2 * z])  # center point

    # calculate the ellipsoid
    bboxl = np.floor(com - radii).clip(0, None).astype(int)
    bboxh = (np.ceil(com + radii) + 1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice, bboxl, bboxh))]
    roiaux = aux[tuple(map(slice, bboxl, bboxh))]
    logrid = *map(np.square, np.ogrid[tuple(
        map(slice, (bboxl - com) / radii, (bboxh - com - 1) / radii, 1j * (bboxh - bboxl)))]),
    dst = (1 - sum(logrid)).clip(0, None)
    mask = dst > roiaux
    roi[mask] = 1
    np.copyto(roiaux, dst, where=mask)

    return out


def get_fixed_geo(mask_scan, tumor_type, gmm_list=[], ellipsoid_model=None, model_name=None):
    gmm_model_tiny = None
    gmm_model_non_tiny = None
    if len(gmm_list) == 1:
        gmm_model_tiny = gmm_list[0]
        gmm_model_non_tiny = gmm_list[0]
    elif len(gmm_list) == 2:
        gmm_model_tiny = gmm_list[0]
        gmm_model_non_tiny = gmm_list[1]

    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros(
        (mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.int8)
    # texture_map = np.zeros((mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z), dtype=np.float16)
    tiny_radius, small_radius, medium_radius, large_radius = 4, 8, 16, 32

    if tumor_type == 'tiny':
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            y = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            z = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            sigma = random.uniform(0.5, 1)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

    if tumor_type == 'small':
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            y = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            z = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            sigma = random.randint(1, 2)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_non_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'medium':
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            y = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            z = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            sigma = random.randint(3, 6)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_non_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == 'large':
        num_tumor = random.randint(1, 3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            y = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            z = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            sigma = random.randint(5, 10)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_non_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    if tumor_type == "mix":
        # tiny
        num_tumor = random.randint(3, 10)
        for _ in range(num_tumor):
            # Tiny tumor
            x = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            y = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            z = random.randint(int(0.75 * tiny_radius), int(1.25 * tiny_radius))
            sigma = random.uniform(0.5, 1)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo

        # small
        num_tumor = random.randint(5, 10)
        for _ in range(num_tumor):
            # Small tumor
            x = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            y = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            z = random.randint(int(0.75 * small_radius), int(1.25 * small_radius))
            sigma = random.randint(1, 2)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_non_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

        # medium
        num_tumor = random.randint(2, 5)
        for _ in range(num_tumor):
            # medium tumor
            x = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            y = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            z = random.randint(int(0.75 * medium_radius), int(1.25 * medium_radius))
            sigma = random.randint(3, 6)

            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_non_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste medium tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

        # large
        num_tumor = random.randint(1, 3)
        for _ in range(num_tumor):
            # Large tumor
            x = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            y = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            z = random.randint(int(0.75 * large_radius), int(1.25 * large_radius))
            sigma = random.randint(5, 10)
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1, 2))
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 2))
            # texture = get_texture((4*x, 4*y, 4*z))
            if model_name is None:
                point = random_select(mask_scan)
            elif model_name == 'gmm':
                point = gmm_select(mask_scan, gmm_model_non_tiny)
            elif model_name == 'ellipsoid':
                point = ellipsoid_select(mask_scan, ellipsoid_model)

            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]
            x_low, x_high = new_point[0] - geo.shape[0] // 2, new_point[0] + geo.shape[0] // 2
            y_low, y_high = new_point[1] - geo.shape[1] // 2, new_point[1] + geo.shape[1] // 2
            z_low, z_high = new_point[2] - geo.shape[2] // 2, new_point[2] + geo.shape[2] // 2

            # paste small tumor geo into test sample
            geo_mask[x_low:x_high, y_low:y_high, z_low:z_high] += geo
            # texture_map[x_low:x_high, y_low:y_high, z_low:z_high] = texture

    geo_mask = geo_mask[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2, enlarge_z // 2:-enlarge_z // 2]
    # texture_map = texture_map[enlarge_x//2:-enlarge_x//2, enlarge_y//2:-enlarge_y//2, enlarge_z//2:-enlarge_z//2]
    geo_mask = (geo_mask * mask_scan) >= 1

    return geo_mask


def Quantify(processed_organ_region, organ_hu_lowerbound, organ_standard_val, outrange_standard_val):
    # Quantify the intensity of differnent part of the organ
    interval = (outrange_standard_val - organ_hu_lowerbound) / 3
    processed_organ_region[(processed_organ_region < (
            organ_hu_lowerbound + interval))] = organ_hu_lowerbound
    processed_organ_region[(processed_organ_region >= (organ_hu_lowerbound + interval)) & (
            processed_organ_region < (organ_hu_lowerbound + 2 * interval))] = organ_hu_lowerbound + interval
    processed_organ_region[(processed_organ_region >= (organ_hu_lowerbound + 2 * interval)) & (
            processed_organ_region < outrange_standard_val)] = organ_hu_lowerbound + 2 * interval

    density_organ_map = processed_organ_region.copy()

    processed_organ_region[processed_organ_region < outrange_standard_val] = organ_standard_val
    processed_organ_region[processed_organ_region == outrange_standard_val] = 1

    processed_organ_region[processed_organ_region == 1] = outrange_standard_val

    density_organ_map[(density_organ_map == outrange_standard_val) & (
            processed_organ_region != outrange_standard_val)] = organ_hu_lowerbound + 2 * interval

    return processed_organ_region, density_organ_map


def get_tumor(volume_scan, mask_scan, tumor_type, texture, hu_processor, edge_advanced_blur, organ_hu_lowerbound,
              outrange_standard_val, density_organ_map, gmm_list=[], ellipsoid_model=None, model_name=None):
    geo_mask = get_fixed_geo(mask_scan, tumor_type, gmm_list, ellipsoid_model, model_name)

    if hu_processor:
        tumor_region = geo_mask.astype(np.float32)

        volume_scan_type = volume_scan.dtype
        volume_scan = volume_scan.astype(np.float32)
        density_organ_map = density_organ_map.astype(np.float32)

        kernel = (3, 3)
        for z in range(tumor_region.shape[0]):
            tumor_region[z] = cv2.GaussianBlur(tumor_region[z], kernel, 0)

        interval = (outrange_standard_val - organ_hu_lowerbound) / 3
        # deal with the conflict vessel
        vessel_condition = (density_organ_map == outrange_standard_val) & tumor_region.astype(bool)
        # deal with the high intensity tissue
        high_tissue_condition = (density_organ_map == (organ_hu_lowerbound + 2 * interval)) & tumor_region.astype(bool)

        volume_scan[vessel_condition] *= (organ_hu_lowerbound + interval / 2) / outrange_standard_val
        volume_scan[high_tissue_condition] *= (organ_hu_lowerbound + 2 * interval) / outrange_standard_val

        volume_scan = volume_scan.astype(volume_scan_type)

    sigma = np.random.uniform(1, 2)
    if edge_advanced_blur:
        sigma = np.random.uniform(1.0, 2.1)
    difference = np.random.uniform(65, 145)

    # blur the boundary
    geo_blur = gaussian_filter(geo_mask * 1.0, sigma)
    abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    # abnormally = (volume_scan - texture * geo_mask * difference) * mask_scan

    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_full, abnormally_mask


def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture, hu_processor, organ_standard_val,
                   organ_hu_lowerbound, outrange_standard_val, edge_advanced_blur,
                   gmm_list=[], ellipsoid_model=None, model_name=None):
    # for speed_generate_tumor, we only send the liver part into the generate program
    x_start, x_end = np.where(np.any(mask_scan, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask_scan, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask_scan, axis=(0, 1)))[0][[0, -1]]

    # shrink the boundary
    x_start, x_end = max(0, x_start + 1), min(mask_scan.shape[0], x_end - 1)
    y_start, y_end = max(0, y_start + 1), min(mask_scan.shape[1], y_end - 1)
    z_start, z_end = max(0, z_start + 1), min(mask_scan.shape[2], z_end - 1)

    liver_volume = volume_scan[x_start:x_end, y_start:y_end, z_start:z_end]
    liver_mask = mask_scan[x_start:x_end, y_start:y_end, z_start:z_end]

    # input texture shape: 420, 300, 320
    # we need to cut it into the shape of liver_mask
    # for examples, the liver_mask.shape = 286, 173, 46; we should change the texture shape
    x_length, y_length, z_length = x_end - x_start, y_end - y_start, z_end - z_start
    start_x = random.randint(0, texture.shape[
        0] - x_length - 1)  # random select the start point, -1 is to avoid boundary check
    start_y = random.randint(0, texture.shape[1] - y_length - 1)
    start_z = random.randint(0, texture.shape[2] - z_length - 1)
    cut_texture = texture[start_x:start_x + x_length, start_y:start_y + y_length, start_z:start_z + z_length]

    # Quantify the density of the organ
    select_organ_region = np.isin(liver_mask, [1, 2])
    processed_organ_region = liver_volume.copy()
    processed_organ_region[~select_organ_region] = outrange_standard_val
    processed_organ_region[processed_organ_region > outrange_standard_val] = outrange_standard_val

    # Quantify the density of the organ
    processed_organ_region, density_organ_map = Quantify(processed_organ_region, organ_hu_lowerbound,
                                                         organ_standard_val, outrange_standard_val)

    liver_volume, liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture, hu_processor,
                                         edge_advanced_blur, organ_hu_lowerbound, outrange_standard_val,
                                         density_organ_map, gmm_list, ellipsoid_model, model_name)
    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

    return volume_scan, mask_scan

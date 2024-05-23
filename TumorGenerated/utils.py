### Tumor Generateion
import random
import pywt
# from noise import snoise3
# from numba import njit, prange
from opensimplex import OpenSimplex
import cv2
import elasticdeform
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_tv_chambolle


def wavelet_filter(data, wavelet='db1', level=1):
    coeffs = pywt.wavedecn(data, wavelet, mode='periodization', level=level)
    coeffs_filtered = [coeffs[0]]  # Approximation coefficients remain unchanged

    # Apply thresholding to detail coefficients
    for detail_level in coeffs[1:]:
        filtered_detail = {key: pywt.threshold(value, np.std(value), mode='soft') for key, value in
                           detail_level.items()}
        coeffs_filtered.append(filtered_detail)

    filtered_data = pywt.waverecn(coeffs_filtered, wavelet, mode='periodization')
    return filtered_data


def apply_median_filter(image, size):
    return median_filter(image, size=size)


def generate_simplex_noise(mask_shape, freq=0.05, octaves=4, persistence=0.5, seed=None):
    x, y, z = np.mgrid[:mask_shape[0], :mask_shape[1], :mask_shape[2]]
    x = x.astype(float) / (mask_shape[0] * freq)
    y = y.astype(float) / (mask_shape[1] * freq)
    z = z.astype(float) / (mask_shape[2] * freq)

    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    simplex = OpenSimplex(seed=seed)

    noise = np.zeros(mask_shape, dtype=float)
    freq_mult = 1.0
    amp_mult = 1.0

    for _ in range(octaves):
        noise += amp_mult * np.vectorize(simplex.noise3)(
            x * freq_mult,
            y * freq_mult,
            z * freq_mult
        )
        freq_mult *= 2
        amp_mult *= persistence

    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob, tumor_value=2, background_value=0):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # Add Salt noise (tumor_value)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = tumor_value

    # Add Pepper noise (background_value)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = background_value

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
def get_predefined_texture_old(mask_shape, sigma_a, sigma_b):
    # uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    a_2 = gaussian_filter(a, sigma=sigma_a)
    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2)) + base

    # sample once
    random_sample = np.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter
    b = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (np.sum(b * threshold_mask) / threshold_mask.sum())
    Bj = np.clip(beta*b, 0, 1) # 目前是0-1区间

    return Bj


# here we want to get predefined texutre:
def get_predefined_texture(mask_shape, sigma_a, sigma_b):
    # Step 1: Uniform noise generate
    a = np.random.uniform(0, 1, size=(mask_shape[0], mask_shape[1], mask_shape[2]))
    # a = generate_simplex_noise(mask_shape, seed=42)

    # Step 2: Nonlinear diffusion filtering
    a_denoised = denoise_tv_chambolle(a, weight=0.1, multichannel=False)

    # Step 3: Wavelet transform
    coeffs = pywt.wavedec2(a_denoised, wavelet='haar', level=2)
    coeffs[1:] = [tuple(np.zeros_like(v) for v in coeff) for coeff in coeffs[1:]]
    a_wavelet_denoised = pywt.waverec2(coeffs, wavelet='haar')

    # Normalize to 0-1
    a_wavelet_denoised = (a_wavelet_denoised - np.min(a_wavelet_denoised)) / (np.max(a_wavelet_denoised) - np.min(a_wavelet_denoised))

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


def get_absolute_coordinates(relative_coordinates, original_shape, target_volume, start):
    x_ratio = original_shape[0] / target_volume[0]
    y_ratio = original_shape[1] / target_volume[1]
    z_ratio = original_shape[2] / target_volume[2]

    absolute_x = relative_coordinates[0] * x_ratio
    absolute_y = relative_coordinates[1] * y_ratio
    absolute_z = relative_coordinates[2] * z_ratio

    absolute_x += start[0]
    absolute_y += start[1]
    absolute_z += start[2]

    return np.array([absolute_x, absolute_y, absolute_z], dtype=float)


def gmm_select(mask_scan, gmm_model=None, max_attempts=600):
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
        if any(coord < -10 for coord in potential_point):
            loop_count += 1
            continue
        potential_point = get_absolute_coordinates(potential_point, liver_mask.shape, target_volume,
                                                    start)
        potential_point = np.clip(potential_point, 0, np.array(mask_scan.shape) - 1).astype(int)
        if mask_scan[tuple(potential_point)] == 1:
            # Check if the point is not at the edge
            if not is_edge_point(mask_scan, potential_point):
                return potential_point

        loop_count += 1

    potential_point = random_select(mask_scan)
    return potential_point

def ellipsoid_select(mask_scan, ellipsoid_model=None, max_attempts=600):
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
        if any(coord < -10 for coord in potential_point):
            loop_count += 1
            continue
        potential_point = get_absolute_coordinates(potential_point, liver_mask.shape, target_volume,
                                                    start)
        potential_point = np.clip(potential_point, 0, np.array(mask_scan.shape) - 1).astype(int)
        if mask_scan[tuple(potential_point)] == 1:
            # Check if the point is not at the edge
            if not is_edge_point(mask_scan, potential_point):
                return potential_point

        loop_count += 1

    potential_point = random_select(mask_scan)
    return potential_point

def is_edge_point(mask_scan, potential_points, neighborhood_size=(3, 3, 3), threshold=5):
    # Define the boundaries of the neighborhood around the potential point
    min_bounds = np.maximum(potential_points - np.array(neighborhood_size) // 2, 0)
    max_bounds = np.minimum(potential_points + np.array(neighborhood_size) // 2, np.array(mask_scan.shape) - 1)

    # Extract the neighborhood volume from the mask scan
    neighborhood_volume = mask_scan[min_bounds[0]:max_bounds[0] + 1,
                          min_bounds[1]:max_bounds[1] + 1,
                          min_bounds[2]:max_bounds[2] + 1]

    # Count the number of liver voxels in the neighborhood
    liver_voxel_count = np.sum(neighborhood_volume == 1)

    # Check if the liver voxel count is below the threshold
    if liver_voxel_count < threshold:
        return True  # Potential point is considered to be on the edge
    else:
        return False  # Potential point is considered to be inside the liver


# Step 2 : generate the ellipsoid
def get_ellipsoid(x, y, z):
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


def get_tumor(volume_scan, mask_scan, tumor_type, texture, gmm_list=[], ellipsoid_model=None, model_name=None):
    geo_mask = get_fixed_geo(mask_scan, tumor_type, gmm_list, ellipsoid_model, model_name)

    sigma = np.random.uniform(1, 2)
    difference = np.random.uniform(65, 145)

    # blur the boundary
    geo_blur = gaussian_filter(geo_mask * 1.0, sigma)
    abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    # abnormally = (volume_scan - texture * geo_mask * difference) * mask_scan

    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_full, abnormally_mask


def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture, gmm_list=[], ellipsoid_model=None, model_name=None):
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

    liver_volume, liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture, gmm_list,
                                         ellipsoid_model, model_name)
    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

    return volume_scan, mask_scan

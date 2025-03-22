### Tumor Generateion
import random

import cv2
import elasticdeform
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, sobel, gaussian_filter1d
from skimage.restoration import denoise_tv_chambolle


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


def get_predefined_texture_b(mask_shape, sigma_a, sigma_b):
    # Step 1: Uniform noise generation
    a = np.random.uniform(0, 1, size=mask_shape).astype(np.float32)

    # Step 2: Nonlinear diffusion filtering
    a_denoised = denoise_tv_chambolle(a, weight=0.1, multichannel=False)

    # Step 3: 3D Wavelet transform
    coeffs = pywt.wavedecn(a_denoised, wavelet='db4', level=2)
    coeffs[1] = {k: 0.3 * v for k, v in coeffs[1].items()}  # 只保留低层高频信息
    for i in range(2, len(coeffs)):  # 直接遍历后续层，避免创建额外变量
        coeffs[i] = {k: np.zeros_like(v) for k, v in coeffs[i].items()}
    a_wavelet_denoised = pywt.waverecn(coeffs, wavelet='db4')

    # Normalize to 0-1
    min_val, max_val = np.min(a_wavelet_denoised), np.max(a_wavelet_denoised)
    a_wavelet_denoised = (a_wavelet_denoised - min_val) / (max_val - min_val + 1e-6)  # 避免除0错误

    # Step 4: Optimized Gaussian filter (using separable filtering)
    a_2 = a_wavelet_denoised
    for ax in range(3):  # 分别沿 x, y, z 方向滤波，减少计算量
        a_2 = gaussian_filter1d(a_2, sigma=sigma_a, axis=ax, mode='nearest')

    scale = np.random.uniform(0.19, 0.21)
    base = np.random.uniform(0.04, 0.06)
    a = scale * (a_2 - np.min(a_2)) / (np.max(a_2) - np.min(a_2) + 1e-6) + base

    # Sample once
    random_sample = np.random.uniform(0, 1, size=mask_shape).astype(np.float32)
    b = (a > random_sample).astype(np.float32)

    # Apply Gaussian filter using separable filtering
    from joblib import Parallel, delayed
    def apply_filter(axis):
        return gaussian_filter1d(b, sigma=sigma_b, axis=axis, mode='nearest')
    results = Parallel(n_jobs=3)(delayed(apply_filter)(ax) for ax in range(3))
    b = np.mean(results, axis=0) 

    # Scaling and clipping
    u_0 = np.random.uniform(0.5, 0.55)
    threshold_mask = b > 0.12
    threshold_sum = np.sum(threshold_mask)
    beta = u_0 / (np.sum(b * threshold_mask) / (threshold_sum + 1e-6))  # 避免除 0
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

def ellipsoid_select(mask_scan, ellipsoid_model=None, max_attempts=600, edge_op="both"):
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


def check_sobel(mask_scan, potential_point, sobel_threshold=405, sobel_neighborhood_size=(7, 7, 7)):
    # Calculate the neighborhood bounds, ensuring it stays within the mask scan limits
    min_bounds = np.maximum(potential_point - np.array(sobel_neighborhood_size) // 2, 0)
    max_bounds = np.minimum(potential_point + np.array(sobel_neighborhood_size) // 2, np.array(mask_scan.shape) - 1)

    # Extract the neighborhood sub-region
    neighborhood_sn = mask_scan[min_bounds[0]:max_bounds[0] + 1,
                                min_bounds[1]:max_bounds[1] + 1,
                                min_bounds[2]:max_bounds[2] + 1]

    # Compute Sobel filters only on the neighborhood region
    sobel_x = sobel(neighborhood_sn, axis=0)
    sobel_y = sobel(neighborhood_sn, axis=1)
    sobel_z = sobel(neighborhood_sn, axis=2)

    # Calculate the gradient magnitude of the Sobel filter
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    # Get the gradient magnitude at the central point of the neighborhood
    central_point = np.array(sobel_neighborhood_size) // 2
    gradient_value = gradient_magnitude[tuple(central_point)]

    # Return True if the gradient magnitude exceeds the threshold
    return gradient_value > sobel_threshold

def is_edge_point(mask_scan, potential_point, edge_op="both", neighborhood_size=(3, 3, 3), volume_threshold=5,
                  sobel_threshold=405):
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

    # 根据选择的操作模式判断是否为边缘点
    if edge_op == "volume":
        return check_volume()
    elif edge_op == "sobel":
        return check_sobel(mask_scan, potential_point, sobel_threshold)
    elif edge_op == "any":
        return check_volume() and check_sobel(mask_scan, potential_point, sobel_threshold)
    elif edge_op == "both":
        return check_volume() or check_sobel(mask_scan, potential_point, sobel_threshold)
    elif edge_op == "none":
        return False
    else:
        raise ValueError("Invalid edge_op option. Choose from 'volume', 'sobel', 'any', 'both' or 'none'.")


def get_ellipsoid(x, y, z):
    """"
    x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    """
    # 将浮点数值转换为整数（添加安全转换）
    x = int(round(float(x)))
    y = int(round(float(y)))
    z = int(round(float(z)))
    
    # 确保最小尺寸为1
    x = max(1, x)
    y = max(1, y)
    z = max(1, z)
    
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


def get_fixed_geo(mask_scan, tumor_type, ellipsoid_model=None):
    enlarge_x, enlarge_y, enlarge_z = 160, 160, 160
    geo_mask = np.zeros(
        (mask_scan.shape[0] + enlarge_x, mask_scan.shape[1] + enlarge_y, mask_scan.shape[2] + enlarge_z),
        dtype=np.uint8)

    # 优化后的肿瘤参数配置
    tumor_config = {
        'tiny':    {'base_radius': 6,  'num_range': (2,5),  'sigma': (0.8,1.5), 'vol_range':(3.4e3, 1.6e4)},
        'small':   {'base_radius': 10, 'num_range': (1,4),  'sigma': (1.5,3),   'vol_range':(1.6e4, 4e4)},
        'medium':  {'base_radius': 18, 'num_range': (1,3),  'sigma': (3,5),     'vol_range':(4e4, 1e5)},
        'large':   {'base_radius': 25, 'num_range': (1,2),  'sigma': (5,8),     'vol_range':(1e5, 1.1e5)},
        'mix':     {'components': [
                            {'type':'tiny',  'prob':0.3},
                            {'type':'small', 'prob':0.4},
                            {'type':'medium','prob':0.2},
                            {'type':'large', 'prob':0.1}
                        ]}
    }

    def create_tumor(config):
        # 根据体积范围动态计算半径
        target_vol = np.random.uniform(*config['vol_range'])
        radius_factor = (target_vol * 3/(4*np.pi)) ** (1/3)
        
        num_tumor = random.randint(*config['num_range'])
        for _ in range(num_tumor):
            # 添加参数约束
            x = max(1, np.random.normal(radius_factor, radius_factor*0.3))
            y = max(1, np.random.normal(radius_factor, radius_factor*0.3)) 
            z = max(1, np.random.normal(radius_factor*0.8, radius_factor*0.2))
            
            sigma = random.uniform(*config['sigma'])
            geo = get_ellipsoid(x, y, z)
            geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0, 1, 2))

            point = ellipsoid_select(mask_scan, ellipsoid_model) if ellipsoid_model is not None else random_select(
                mask_scan)
            new_point = [point[0] + enlarge_x // 2, point[1] + enlarge_y // 2, point[2] + enlarge_z // 2]

            # 添加边界检查确保切片范围有效
            x_low = max(0, new_point[0] - geo.shape[0] // 2)
            x_high = min(geo_mask.shape[0], new_point[0] + geo.shape[0] // 2)
            y_low = max(0, new_point[1] - geo.shape[1] // 2)
            y_high = min(geo_mask.shape[1], new_point[1] + geo.shape[1] // 2)
            z_low = max(0, new_point[2] - geo.shape[2] // 2)
            z_high = min(geo_mask.shape[2], new_point[2] + geo.shape[2] // 2)

            # 调整geo数组大小以匹配目标区域
            actual_geo = geo[:x_high-x_low, :y_high-y_low, :z_high-z_low]
            
            np.add.at(geo_mask, (slice(x_low, x_high), slice(y_low, y_high), slice(z_low, z_high)), actual_geo)

    # 修改调用逻辑
    if tumor_type == 'mix':
        for comp in tumor_config['mix']['components']:
            if random.random() < comp['prob']:
                create_tumor(tumor_config[comp['type']])
    else:
        create_tumor(tumor_config[tumor_type])

    geo_mask = geo_mask[enlarge_x // 2:-enlarge_x // 2, enlarge_y // 2:-enlarge_y // 2, enlarge_z // 2:-enlarge_z // 2]
    geo_mask = (geo_mask * mask_scan) >= 1  # Apply mask to geo_mask

    return geo_mask


def get_tumor(volume_scan, mask_scan, tumor_type, texture, edge_advanced_blur=False, ellipsoid_model=None):
    geo_mask = get_fixed_geo(mask_scan, tumor_type, ellipsoid_model)

    sigma = np.random.uniform(1, 2)
    if edge_advanced_blur:
        sigma = np.random.uniform(1.0, 2.1)
    base_diff = np.random.normal(65, 15)  # 正态分布代替均匀分布
    difference = base_diff * (0.8 + 0.4*texture)  # 引入纹理相关性

    # blur the boundary
    geo_blur = gaussian_filter(geo_mask * 1.0, sigma)
    abnormally = (volume_scan - texture * geo_blur * difference) * mask_scan
    # abnormally = (volume_scan - texture * geo_mask * difference) * mask_scan

    abnormally_full = volume_scan * (1 - mask_scan) + abnormally
    abnormally_mask = mask_scan + geo_mask

    return abnormally_full, abnormally_mask


def SynthesisTumor(volume_scan, mask_scan, tumor_type, texture, edge_advanced_blur, ellipsoid_model=None):
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

    liver_volume, liver_mask = get_tumor(liver_volume, liver_mask, tumor_type, cut_texture,
                                         edge_advanced_blur, ellipsoid_model)
    volume_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_volume
    mask_scan[x_start:x_end, y_start:y_end, z_start:z_end] = liver_mask

    return volume_scan, mask_scan

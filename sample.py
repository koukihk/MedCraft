import os
import numpy as np
from scipy import ndimage
import nibabel as nib


def voxel2mm(N, spacing):
    return int(N * np.prod(spacing))


def size2r(size):
    r = np.cbrt(3 * size / (4 * np.pi))
    return r


all_r = []
# [0-5) [5-10) [10-15) [15-20) >= 20
interval_count = [0, 0, 0, 0, 0]
hit_count = [0, 0, 0, 0, 0]

for file in files:
    gt = nib.load(os.path.join(gt_root, file)).get_fdata()
    pred = nib.load(os.path.join(pred_root, file)).get_fdata()

    gt = (gt == 2)
    pred = (pred == 2)

    labels, nb = ndimage.label(gt)
    print(file, nb)
    for idx in range(1, nb + 1):
        component = (labels == idx)
        pixel_sum = np.sum(component)
        size = voxel2mm(pixel_sum, (1, 1, 1))
        r = size2r(size)

        if (pixel_sum > 20):  # too small is noise
            size = voxel2mm(pixel_sum, (1, 1, 1))
            r = size2r(size)
            all_r.append(r)

            pred_hit = np.sum(np.logical_and(pred, component)) > 0  # check if prediction hit this component
            # check r
            if r < 5:
                pos = 0
            elif r >= 5 and r < 10:
                pos = 1
            elif r >= 10 and r < 15:
                pos = 2
            elif r >= 15 and r < 20:
                pos = 3
            else:
                pos = 4

            interval_count[pos] += 1
            hit_count[pos] += pred_hit

            print('[{} {:.1f} {}]'.format(size, r, pred_hit), end='\t')
    print('\n')

print("all count", interval_count)
print("hit", hit_count)
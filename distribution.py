import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage,stats


def analyze_tumor_location(label_data):
    labeled_components, num_components = ndimage.label(label_data == 2)
    np.save("labeled_components.npy", labeled_components)

    tumor_positions = []

    for i in range(1, num_components + 1):
        bounding_box = ndimage.find_objects(labeled_components == i)[0]
        center_coords = [(slice.start + slice.stop - 1) / 2 for slice in bounding_box]
        tumor_positions.append(tuple(center_coords))

    return labeled_components, tumor_positions

def process_all_cts(data_folder):
    ct_files = sorted(os.listdir(os.path.join(data_folder, "img")))
    result_list = []

    for ct_file in ct_files:
        if ct_file.startswith("._"):
            continue
        img_path = os.path.join(data_folder, "img", ct_file)
        label_path = os.path.join(data_folder, "label", ct_file)

        if not (os.path.isfile(img_path) and os.path.isfile(label_path)):
            continue

        label_data = nib.load(label_path).get_fdata()
        labeled_components, tumor_positions = analyze_tumor_location(label_data)

        if tumor_positions:
            result_list.append({"CT_File": ct_file, "Tumor_Positions": tumor_positions})

    result_df = pd.DataFrame(result_list)

    return result_df

data_folder = "datafolds/04_LiTS"

result_df = process_all_cts(data_folder)

tumor_positions = []

for _, row in result_df.iterrows():
    for tumor_pos in row['Tumor_Positions']:
        tumor_positions.append({'x': tumor_pos[0], 'y': tumor_pos[1], 'z': tumor_pos[2]})

tumor_data = pd.DataFrame(tumor_positions)

mean_x, std_x = stats.norm.fit(tumor_data['x'])
mean_y, std_y = stats.norm.fit(tumor_data['y'])
mean_z, std_z = stats.norm.fit(tumor_data['z'])

print(mean_x, std_x, mean_y, std_y, mean_z, std_z)
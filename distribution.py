import os
from datetime import datetime
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from scipy import ndimage


def analyze_tumor_location(label_data):
    labeled_components, num_components = ndimage.label(label_data == 2)
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
output_dir = f"distribution/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

result_df = process_all_cts(data_folder)
tumor_coordinates = []

for _, row in result_df.iterrows():
    for tumor_pos in row['Tumor_Positions']:
        tumor_coordinates.append({'x': tumor_pos[0], 'y': tumor_pos[1], 'z': tumor_pos[2]})

tumor_data = pd.DataFrame(tumor_coordinates)

plt.figure(figsize=(12, 8))
plt.hist2d(tumor_data['x'].values, tumor_data['y'].values, bins=(30, 30), cmap='viridis', cmin=1, alpha=0.7)
plt.title('2D Heatmap of Tumor Coordinates in x-y Plane')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')

plt.scatter(tumor_data['x'].values, tumor_data['y'].values, marker='.', color='red', alpha=0.1)

plt.xlim(min(tumor_data['x'].values), max(tumor_data['x'].values))
plt.ylim(min(tumor_data['y'].values), max(tumor_data['y'].values))

heatmap = os.path.join(output_dir, "overall_tumor_distribution.png")
plt.savefig(heatmap)

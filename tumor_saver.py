import os
import re
import nibabel as nib

class TumorSaver:
    @staticmethod
    def get_datatype(datatype):
        datatype_map = {
            2: 'uint8',
            4: 'int16',
            8: 'int32',
            16: 'float32',
            32: 'complex64',
            64: 'float64'
        }
        return datatype_map.get(datatype.item(), 'uint8')

    @staticmethod
    def get_new_filename(existing_files, base_filename):
        base, ext = os.path.splitext(base_filename)
        pattern = re.compile(r'^(\d+_)?(\d+)_\w*\.nii\.gz$')
        counts = []

        for file in existing_files:
            match = pattern.match(file)
            if match:
                count = int(match.group(2))
                counts.append(count)

        if not counts:
            return f"synt_{base_filename}"  # No conflicts, add 'synt_' prefix

        max_count = max(counts)
        new_count = max_count + 1

        new_filename = f"synt_{new_count}_{base}{ext}"  # Add 'synt_' prefix and numbering
        return new_filename

    @staticmethod
    def save_nifti(data, affine_matrix, data_type, output_path, filename):
        if os.path.exists(output_path):
            existing_files = os.listdir(output_path)
            new_filename = TumorSaver.get_new_filename(existing_files, filename)
        else:
            os.makedirs(output_path, exist_ok=True)
            new_filename = f"synt_{filename}"

        output_file = os.path.join(output_path, new_filename)
        nib.save(
            nib.Nifti1Image(data.astype(data_type), affine_matrix),
            output_file
        )
        print(f"Saved {output_file}")

    @staticmethod
    def save_data(d, folder='global'):
        image_data_type = TumorSaver.get_datatype(d['image_meta_dict']['datatype'])
        image_affine_matrix = d['image_meta_dict']['original_affine'][0]

        label_data_type = TumorSaver.get_datatype(d['label_meta_dict']['datatype'])
        label_affine_matrix = d['label_meta_dict']['original_affine'][0]

        image = d['image'][0].squeeze(0).cpu().numpy()
        label = d['label'][0].squeeze(0).cpu().numpy()

        image_outputs = f'synt/{folder}/image'
        label_outputs = f'synt/{folder}/label'

        image_filename = os.path.basename(d['image_meta_dict']['filename_or_obj'][0])
        label_filename = os.path.basename(d['label_meta_dict']['filename_or_obj'][0])

        TumorSaver.save_nifti(image, image_affine_matrix, image_data_type, image_outputs, image_filename)
        TumorSaver.save_nifti(label, label_affine_matrix, label_data_type, label_outputs, label_filename)

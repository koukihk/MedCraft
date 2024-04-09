import nibabel as nib


def check_nii_metadata(file_path):
    try:
        img = nib.load(file_path)

        print("Header information:")
        print(img.header)

        print("Image shape:", img.shape)

        print("Voxel dimensions:", img.header.get_zooms())

        print("Data type:", img.header.get_data_dtype())

    except Exception as e:
        print("An error occurred:", e)


import nibabel as nib


def compare_nii_metadata(file_path1, file_path2):
    try:
        img1 = nib.load(file_path1)
        img2 = nib.load(file_path2)

        if img1.header != img2.header:
            print("Metadata is different:")
            print("File 1 metadata:")
            print(img1.header)
            print("File 2 metadata:")
            print(img2.header)
        else:
            print("Metadata is identical.")

        if img1.shape != img2.shape:
            print("Image shapes are different:")
            print("File 1 shape:", img1.shape)
            print("File 2 shape:", img2.shape)
        else:
            print("Image shapes are identical.")

        if img1.header.get_zooms() != img2.header.get_zooms():
            print("Voxel dimensions are different:")
            print("File 1 voxel dimensions:", img1.header.get_zooms())
            print("File 2 voxel dimensions:", img2.header.get_zooms())
        else:
            print("Voxel dimensions are identical.")

        if img1.header.get_data_dtype() != img2.header.get_data_dtype():
            print("Data types are different:")
            print("File 1 data type:", img1.header.get_data_dtype())
            print("File 2 data type:", img2.header.get_data_dtype())
        else:
            print("Data types are identical.")

    except Exception as e:
        print("An error occurred:", e)


file_path1 = "/share/home/ncu_418000230020/SyntheticTumors/synt/medium/image/synt_liver_91.nii.gz"
file_path2 = "/share/home/ncu_418000230020/SyntheticTumors/datafolds/04_LiTS/img/liver_91.nii.gz"
compare_nii_metadata(file_path1, file_path2)
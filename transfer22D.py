import os
import numpy as np
import nibabel as nib
import imageio
import cv2


def read_niifile(niifilepath):  # 读取niifile文件
    img = nib.load(niifilepath)  # 提取niifile文件
    img_fdata = img.get_fdata(dtype='float32')
    return img_fdata


def save_fig(niifilepath, savepath, num, name):  # 保存为图片
    name = name.split('-')[1]
    filepath_seg = niifilepath + "segmentation\\" + "segmentation-" + name
    filepath_vol = niifilepath + "volume\\" + "volume-" + name
    savepath_seg = savepath + "segmentation\\"
    savepath_vol = savepath + "volume\\"

    if not os.path.exists(savepath_seg):
        os.makedirs(savepath_seg)
    if not os.path.exists(savepath_vol):
        os.makedirs(savepath_vol)

    fdata_vol = read_niifile(filepath_vol)
    fdata_seg = read_niifile(filepath_seg)
    (x, y, z) = fdata_seg.shape
    total = x * y

    for k in range(z):
        silce_seg = fdata_seg[:, :, k]  # 三个位置表示三个不同角度的切片
        if silce_seg.max() == 0:
            continue
        else:
            silce_seg = (silce_seg - silce_seg.min()) / (silce_seg.max() - silce_seg.min()) * 255
            silce_seg = cv2.threshold(silce_seg, 1, 255, cv2.THRESH_BINARY)[1]

            if (np.sum(silce_seg == 255) / total) > 0.015:
                silce_vol = fdata_vol[:, :, k]
                silce_vol = (silce_vol - silce_vol.min()) / (silce_vol.max() - silce_vol.min()) * 255

                imageio.imwrite(os.path.join(savepath_seg, '{}.png'.format(num)), silce_seg)
                imageio.imwrite(os.path.join(savepath_vol, '{}.png'.format(num)), silce_vol)
                num += 1
        # 将切片信息保存为png格式
    return num


if __name__ == '__main__':

    path = 'E:\\dataset\\LiTS17\\'
    savepath = 'E:\\dataset\\LiTS17\\2d\\'
    filenames = os.listdir(path + "segmentation")
    num = 0
    for filename in filenames:
        num = save_fig(path, savepath, num, filename)
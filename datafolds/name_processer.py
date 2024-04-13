import os
import random

import argparse

parser = argparse.ArgumentParser(description='brats21 segmentation testing')

parser.add_argument('--isEarly', action='store_true')
parser.add_argument('--gen_folder', default='normal')
parser.add_argument('--txt_id', default='0')


def process_labelname(labelname):
    # 匹配规则
    if 'lits-liver' in labelname:
        return labelname.replace('synt_lits-liver_', 'synt_liver_')
    elif 'chaos-' in labelname:
        return labelname.replace('synt_chaos-', 'synt_').replace('_segmentation', '_image')
    elif 'tcia-label' in labelname:
        return labelname.replace('synt_tcia-label', 'synt_PANCREAS_')
    elif 'multi-atlas-label' in labelname:
        return labelname.replace('synt_multi-atlas-label', 'synt_img')
    else:
        return labelname

def main():
    args = parser.parse_args()

    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    label_type = 'noearly'
    if args.isEarly:
        label_type = 'early'
    file_path = f'/share/home/ncu22/SyntheticTumors/synt/{args.gen_folder}/{label_type}_tumor_label/{label_type}_valid_names.txt'
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    imagenames = []
    labelnames = []

    # 提取imagename和labelname
    for line in lines:
        labelname = os.path.basename(line.strip())  # 获取文件名部分
        imagename = process_labelname(labelname)
        imagenames.append(imagename)
        labelnames.append(labelname)

    # 随机打乱顺序
    combined = list(zip(imagenames, labelnames))
    random.shuffle(combined)
    imagenames[:], labelnames[:] = zip(*combined)

    # 分割成训练集和验证集
    total_count = len(imagenames)
    train_count = min(80, total_count)  # 确保不超过总数和最小数
    val_count = min(20, total_count)    # 确保不超过总数和最小数

    train_imagenames = imagenames[:train_count]
    train_labelnames = labelnames[:train_count]
    val_imagenames = imagenames[train_count:train_count+val_count]
    val_labelnames = labelnames[train_count:train_count+val_count]

    # 写入训练集文件
    with open(f'/share/home/ncu22/DiffTumor/STEP2.DiffusionModel/cross_eval/liver_tumor_data_{label_type}_fold/real_tumor_train_{args.txt_id}.txt', 'w') as train_file:
        for img, label in zip(train_imagenames, train_labelnames):
            train_file.write(f'{img}\t\t{label}\n')

    # 写入验证集文件
    with open(f'/share/home/ncu22/DiffTumor/STEP2.DiffusionModel/cross_eval/liver_tumor_data_{label_type}_fold/real_tumor_val_{args.txt_id}.txt', 'w') as val_file:
        for img, label in zip(val_imagenames, val_labelnames):
            val_file.write(f'{img}\t\t{label}\n')

if __name__ == "__main__":
    main()

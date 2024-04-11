import os
import random

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

def main(file_path):
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
    with open('real_tumor_train.txt', 'w') as train_file:
        for img, label in zip(train_imagenames, train_labelnames):
            train_file.write(f'{img}\t\t{label}\n')

    # 写入验证集文件
    with open('real_tumor_val.txt', 'w') as val_file:
        for img, label in zip(val_imagenames, val_labelnames):
            val_file.write(f'{img}\t\t{label}\n')

if __name__ == "__main__":
    file_path = '/share/home/ncu22/SyntheticTumors/synt/normal/noearly_tumor_label/noearly_valid_names.txt'
    main(file_path)

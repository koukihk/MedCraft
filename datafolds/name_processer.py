import os
import random
import argparse

parser = argparse.ArgumentParser(description='brats21 segmentation testing')

parser.add_argument('--isEarly', action='store_true')
parser.add_argument('--gen_folder', default='normal')
parser.add_argument('--txt_id', default='0')


def process_labelname(labelname):
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

    with open(file_path, 'r') as file:
        lines = file.readlines()

    total_count = len(lines)

    if total_count < 100:
        raise ValueError("Total count of samples is less than 100. Unable to proceed.")

    # Shuffle the lines
    random.shuffle(lines)

    # Split into train and validation sets
    train_lines = lines[:80]
    val_lines = lines[80:100]

    train_imagenames = []
    train_labelnames = []
    val_imagenames = []
    val_labelnames = []

    for line in train_lines:
        labelname = os.path.basename(line.strip())
        imagename = process_labelname(labelname)
        train_imagenames.append(imagename)
        train_labelnames.append(labelname)

    for line in val_lines:
        labelname = os.path.basename(line.strip())
        imagename = process_labelname(labelname)
        val_imagenames.append(imagename)
        val_labelnames.append(labelname)

    with open(
            f'/share/home/ncu22/DiffTumor/STEP2.DiffusionModel/cross_eval/liver_tumor_data_{label_type}_fold/real_tumor_train_{args.txt_id}.txt',
            'w') as train_file:
        for img, label in zip(train_imagenames, train_labelnames):
            train_file.write(f'{img}\t\t{label}\n')

    with open(
            f'/share/home/ncu22/DiffTumor/STEP2.DiffusionModel/cross_eval/liver_tumor_data_{label_type}_fold/real_tumor_val_{args.txt_id}.txt',
            'w') as val_file:
        for img, label in zip(val_imagenames, val_labelnames):
            val_file.write(f'{img}\t\t{label}\n')


if __name__ == "__main__":
    main()

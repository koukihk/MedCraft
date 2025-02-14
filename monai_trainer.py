import json
import os
import shutil
import sys

# import tempfile
# import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
# import json
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler  # native AMP
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../../pipextra/lib/python3.6/site-packages')  # add missing packages


def json_get_fold(datalist, basedir, fold=0, key='training'):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if 'fold' in d and d['fold'] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


import math


# template copied from torch.utils.data.distributed.DistributedSampler
class AMDistributedSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.shuffle = shuffle
        self.make_even = make_even

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # to track of smaller batches
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        # deterministically shuffle based on epoch

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible (otherwise will return last batch smaller)

        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(
                        indices))  # this ensures we get valid ids (if dataset is much smaller then world_size
                    indices += [indices[ids] for ids in extra_ids]

            assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def distributed_all_gather(tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False,
                           is_valid=None):
    if world_size is None:
        world_size = torch.distributed.get_world_size()

    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)  # it can't be more then world_size
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)

    if not no_barrier:
        torch.distributed.barrier()  # synch processess, do we need it??

    tensor_list_out = []
    with torch.no_grad():  # ? do we need it

        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]  # list of bools
            # print('is_valid list', is_valid)

        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)

            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]  # keep only valid elements
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
                # print('updated gather list', gather_list)

            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]  # convert to numpy

            tensor_list_out.append(gather_list)

    return tensor_list_out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n

        self.count += n
        # self.avg = self.sum / self.count if self.count > 0 else self.sum
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

import numpy as np
from scipy.ndimage import label

def denoise_pred(pred: np.ndarray):
    """
    # pred.shape can be either (3, H, W, D) or (batch_size, 3, H, W, D)
    # 0: background, 1: liver, 2: tumor.
    """
    if pred.ndim == 4:
        return denoise_single_sample(pred)
    elif pred.ndim == 5:
        batch_denoised = np.zeros_like(pred)
        for i in range(pred.shape[0]):
            batch_denoised[i, ...] = denoise_single_sample(pred[i, ...])
        return batch_denoised
    else:
        raise ValueError(f"Unexpected input dimensions: {pred.shape}")

def denoise_single_sample(pred: np.ndarray):
    denoise_pred = np.zeros_like(pred)

    live_channel = pred[1, ...]
    labels, nb = label(live_channel)
    max_sum = -1
    choice_idx = -1
    for idx in range(1, nb + 1):
        component = (labels == idx)
        if np.sum(component) > max_sum:
            choice_idx = idx
            max_sum = np.sum(component)
    component = (labels == choice_idx)
    denoise_pred[1, ...] = component

    liver_dilation = ndimage.binary_dilation(denoise_pred[1, ...], iterations=30).astype(bool)
    denoise_pred[2, ...] = pred[2, ...].astype(bool) * liver_dilation

    denoise_pred[0, ...] = 1 - np.logical_or(denoise_pred[1, ...], denoise_pred[2, ...])

    return denoise_pred

def calculate_quality_proportion(segmentation_output, tumor_mask):
    tumor_mask = (tumor_mask == 2).float()
    tumor_voxels = tumor_mask.sum().item()
    if tumor_voxels == 0:
        return 0  # Avoid division by zero
    if segmentation_output.ndim == 5:
        seg_tumor_prob = segmentation_output[:, 2, ...]
    else:
        seg_tumor_prob = segmentation_output[2, ...]
    matched_voxels = (seg_tumor_prob * tumor_mask).sum().item()
    return matched_voxels / tumor_voxels


def filter_synthetic_tumor_batch(data, target, model, model_inferer, use_inferer, threshold=0.5):
    filtered_data = []
    filtered_target = []

    for i in range(data.size(0)):
        single_data = data[i].unsqueeze(0)
        single_target = target[i].unsqueeze(0)

        with torch.no_grad():
            output = model_inferer(single_data) if use_inferer else model(single_data)
            # output = torch.sigmoid(output)
            output = denoise_pred(output.detach().cpu().numpy())
            output = torch.tensor(output, device=data.device)

            quality_proportion = calculate_quality_proportion(output, single_target)
            if quality_proportion >= threshold:
                filtered_data.append(single_data)
                filtered_target.append(single_target)
            else:
                print("A data pair has been filtered out.")

    if len(filtered_data) == 0:
        return None, None
    return torch.cat(filtered_data, dim=0), torch.cat(filtered_target, dim=0)


import random
from torch.distributions import Beta
from torch.cuda.amp import autocast
import time
import numpy as np


class MixDataLoader:
    def __init__(self, loader):
        self.loader = loader
        self.data_list = list(loader)

    def get_random_batch(self):
        return random.choice(self.data_list)


def cutmix_3d(data, target, mixup_loader, beta=1.0, cutmix_prob=0.5):
    if np.random.rand() > cutmix_prob:
        return data, target

    random_batch = mixup_loader.get_random_batch()
    other_data = random_batch["image"].cuda(data.device)
    other_target = random_batch["label"].cuda(target.device)

    if other_data.shape != data.shape or other_target.shape != target.shape:
        raise ValueError("CutMix requires both batches to have the same shape.")

    lam = np.random.beta(beta, beta) if beta > 0 else 1

    D, H, W = data.shape[2:]

    cut_d, cut_h, cut_w = int(D * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))

    d1, h1, w1 = np.random.randint(0, D - cut_d), np.random.randint(0, H - cut_h), np.random.randint(0, W - cut_w)

    mixed_data = data.clone()
    mixed_target = target.clone()

    mixed_data[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w] = \
        other_data[:, :, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w]

    mixed_target[:, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w] = \
        other_target[:, d1:d1 + cut_d, h1:h1 + cut_h, w1:w1 + cut_w]

    return mixed_data, mixed_target

def to_one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels.long()]

def mixup_3d(data, target, mixup_loader, alpha=0.4, mixup_prob=0.5):
    if random.random() > mixup_prob:
        return data, target

    if alpha <= 0:
        return data, target

    # Sample lambda from Beta distribution
    beta_dist = Beta(alpha, alpha)
    lam = beta_dist.sample().item()

    # Get random batch from cached loader
    random_batch = mixup_loader.get_random_batch()
    other_data = random_batch["image"].cuda(data.device)
    other_target = random_batch["label"].cuda(target.device)

    # Ensure shapes match
    if other_data.shape != data.shape or other_target.shape != target.shape:
        raise ValueError("Mixup requires both batches to have the same shape.")

    # Apply Mixup
    mixed_data = lam * data + (1 - lam) * other_data
    mixed_target = lam * target + (1 - lam) * other_target

    return mixed_data, mixed_target

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, filter_model=None, filter_inferer=None):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    mixup_loader = MixDataLoader(loader)

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']

        data, target = data.cuda(args.rank), target.cuda(args.rank)

        if args.filter:
            data, target = filter_synthetic_tumor_batch(
                data, target, filter_model, filter_inferer, use_inferer=True, threshold=args.filter_threshold
            )
            if data is None:
                continue

        if args.cutmix:
            data, target = cutmix_3d(
                data, target, mixup_loader,
                beta=args.cutmix_beta,
                cutmix_prob=args.cutmix_prob
            )

        if args.mixup:
            target = to_one_hot(target, num_classes=3)
            data, target = mixup_3d(
                data, target, mixup_loader,
                alpha=args.mixup_alpha,
                mixup_prob=args.mixup_prob
            )

        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)

        if args.rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()

    return run_loss.avg


def resample(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


def val_epoch(model, loader, val_shape_dict, epoch, loss_func, args, model_inferer=None, post_label=None,
              post_pred=None):
    model.eval()
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()

    with torch.no_grad():

        for idx, batch_data in enumerate(loader):

            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']

            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    torch.cuda.empty_cache()
                    logits = model_inferer(data)  # another inferer (e.g. sliding window)
                else:
                    logits = model(data)

            loss = loss_func(logits, target)

            logits = torch.softmax(logits, 1).cpu().numpy()
            logits = np.argmax(logits, axis=1).astype(np.uint8)
            target = target.cpu().numpy()[:, 0, :, :, :]

            name = batch_data["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
            val_shape = val_shape_dict[name]

            pred = resample(logits[0], val_shape)
            y = resample(target[0], val_shape)

            dice_list_sub = []
            for i in range(1, args.num_classes):
                organ_Dice = dice(pred == i, y == i)
                dice_list_sub.append(organ_Dice)

            if args.distributed:
                torch.distributed.barrier()
                gather_list_sub = [[0] * len(dice_list_sub) for _ in range(dist.get_world_size())]
                torch.distributed.all_gather_object(gather_list_sub, dice_list_sub)

                classes_metriclist = []
                for i in range(args.num_classes - 1):
                    class_metric = [s[i] for s in gather_list_sub]
                    classes_metriclist.append(class_metric)
                avg_classes = np.mean(classes_metriclist, 1)
                ave_all = np.mean(avg_classes)
                #                 if not loss.is_cuda:
                loss = loss.cuda(args.rank)

                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)

                run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0)),
                                n=args.batch_size * args.world_size)

                run_acc.update(avg_classes, n=1)

            # If you do not use distributed, this program will raise error.
            else:
                avg_classes = np.array(dice_list_sub)
                run_acc.update(avg_classes, n=args.batch_size)
                run_loss.update(loss.item(), n=args.batch_size)

            # print(args.rank, 'end1')
            if args.rank == 0:
                print('Batch mean: Liver: {}, Tumor: {}, all:{}'.format(avg_classes[0], avg_classes[1],
                                                                        np.mean(avg_classes)))
                print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                      'loss: {:.4f}'.format(run_loss.avg),
                      'acc', run_acc.avg,
                      'acc_avg: {:.4f}'.format(np.mean(run_acc.avg)),
                      'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()

    return run_loss.avg, run_acc.avg


def save_checkpoint(model, epoch, args, filename='model.pt', best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()

    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict
    }

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)


def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 args,
                 val_shape_dict,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 val_channel_names=None,
                 post_label=None,
                 post_pred=None,
                 filter_model=None,
                 filter_inferer=None
                 ):
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)

    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', writer.log_dir)

    scaler = None
    if args.amp:  # new native amp
        scaler = GradScaler()

    val_acc_max = 0.

    for epoch in range(start_epoch, args.max_epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        print(args.rank, time.ctime(), 'Epoch:', epoch)

        epoch_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func,
                                 args=args, filter_model=filter_model, filter_inferer=filter_inferer)

        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))

        if args.rank == 0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)

        b_new_best = False
        val_acc = 0
        if ((epoch + 1) % args.val_every == 0):

            if args.distributed:
                torch.distributed.barrier()  # sync processes

            epoch_time = time.time()
            # torch.cuda.empty_cache()
            val_loss, val_acc = val_epoch(model, val_loader, val_shape_dict, epoch=epoch, loss_func=loss_func,
                                          model_inferer=model_inferer, args=args, post_label=post_label,
                                          post_pred=post_pred)
            if args.rank == 0:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(val_loss),
                      'acc', val_acc, 'time {:.2f}s'.format(time.time() - epoch_time))
                with open("test_online.txt", 'a') as f:
                    print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(val_loss),
                          'acc', val_acc, 'time {:.2f}s'.format(time.time() - epoch_time), file=f)
                if writer is not None:
                    writer.add_scalar('val_loss', val_loss, epoch)
                    writer.add_scalar('val_mean_dice', np.mean(val_acc), epoch)
                    if val_channel_names is not None:
                        for val_channel_ind in range(len(val_channel_names)):
                            if val_channel_ind < val_acc.size:
                                writer.add_scalar(val_channel_names[val_channel_ind], val_acc[val_channel_ind], epoch)

                if np.mean(val_acc) > val_acc_max:
                    print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, np.mean(val_acc)))
                    val_acc_max = np.mean(val_acc)
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max, optimizer=optimizer,
                                        scheduler=scheduler)

        if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
            save_checkpoint(model, epoch, args, best_acc=np.mean(val_acc), filename='model_final.pt')
            if b_new_best:
                print('Copying to model.pt new best model!!!!')
                shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

        if scheduler is not None:
            scheduler.step()

    print('monai_trainer DONE, best  acc', val_acc_max)

    return val_acc_max
"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
from network.photo_wct_batch import PhotoWCT, style_transform, FlatFolderDataset
import cv2

loss_log_file = r'./loss_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=loss_log_file,
    filemode='a');

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deeplabv2.DeepR50V2',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['mapillary'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')

parser.add_argument('--wild_dataset', nargs='*', type=str, default=['imagenet'],
                    help='a list consists of imagenet')

parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--wt_layer', nargs='*', type=int, default=[0,0,0,0,0,0,0],
                    help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
parser.add_argument('--wt_reg_weight', type=float, default=0.6)
parser.add_argument('--relax_denom', type=float, default=0.0)
parser.add_argument('--clusters', type=int, default=50)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--dynamic', action='store_true', default=False)

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')
parser.add_argument('--cov_stat_epoch', type=int, default=5,
                    help='cov_stat_epoch')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Visualize intermediate feature')
parser.add_argument('--use_wtloss', action='store_true', default=False,
                    help='Automatic setting from wt_layer')
parser.add_argument('--use_isw', action='store_true', default=False,
                    help='Automatic setting from wt_layer')

parser.add_argument('--csc', action='store_true', default=True,
                    help='Automatic setting from wt_layer')
parser.add_argument('--csc_epoch', type=int, default=2,
                    help='csc_epoch')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)

for i in range(len(args.wt_layer)):
    if args.wt_layer[i] == 1:
        args.use_wtloss = True
    if args.wt_layer[i] == 2:
        args.use_wtloss = True
        args.use_isw = True


patches_imgs_save_path = r'/extra_disk/CSDS/CSC/patches/city/'
patches_nums_save_path = r'/extra_disk/CSDS/CSC/patches/'

# class_pixel_ratio = torch.tensor([509475373, 84222297, 315475529, 9076846, 12154035, 16964991, 2878741, 7636765,
#                                       219931671, 16032511, 55381159, 16818673, 1863031, 96834205, 3699347, 3251826,
#                                       3221320, 1364516, 5723359], dtype=torch.float)
balanced_ratio = torch.tensor([1/19], dtype=torch.float)

all_class_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

# cls_patches_len = list(np.loadtxt(patches_nums_save_path + 'gta_patches_num.txt'))
# prob_class_pixel_ratio = torch.div(class_pixel_ratio, (torch.sum(class_pixel_ratio) + 1e-10))

rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

cls_mean = [-0.1862, -0.1441, -0.2812, -0.0679, -0.0618, -0.2926, -0.1515, -0.1982, -0.3617, -0.2547, -0.0841, -0.1847,
            -0.0294, -0.3307, -0.0653, -0.0930, -0.1345, -0.0786, -0.1934]
cls_std = [0.4638, 0.4290, 0.6432, 0.3898, 0.3562, 0.6514, 0.5370, 0.5890, 0.7231, 0.5366, 0.8815, 0.7375, 0.5637,
           0.9335, 0.4473, 0.5143, 0.5247, 0.4470, 0.6894]

fmap_block = list()
grad_block = list()


def loss_calc_cosin(pred1, pred2):
    # n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.matmul(pred1, pred2) / (torch.norm(pred1) * torch.norm(pred2))
    return output


def loss_calc_cosin_out(pred1, pred2):
    # n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.abs(1 - (torch.matmul(pred1, pred2) / (torch.norm(pred1) * torch.norm(pred2))))
    return output


def loss_calc_dist(pred1, pred2):
    n, c, h, w = pred1.size()
    pred1 = pred1.view(-1).cuda()
    pred2 = pred2.view(-1).cuda()
    # print(pred1)
    # print(pred2)
    output = torch.sum(torch.abs(pred1 - pred2)) / (h * w * c)
    return output


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)


def sample_cls_patch(train_csc_loader):
    cls_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    num_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for val_idx, data in enumerate(train_csc_loader):
        image, gts, _, aux_gts = data
        image, aux_gts, gt = image.cuda(), aux_gts.cuda(), gts.cuda()
        if aux_gts.dim() == 1:
            aux_gts = gt
        ids_unique = gt.unique()

        for i in ids_unique:
            cls_mask = torch.zeros(gt.shape).cuda(non_blocking=True)
            gt_mask = torch.ones(gt.shape).cuda(non_blocking=True) * 255
            i = i.item()
            if i == 255:
                continue
            if cls_index[i] == 50:
                num_index[i] = 1
                continue
            if num_index.count(1) == len(num_index):
                break
            cls_mask[gt == i] = 1
            gt_mask = gt_mask * (1 - cls_mask)
            mask_img = cls_mask.unsqueeze(1).repeat(1, 3, 1, 1) * image.clone()
            mask_gt = cls_mask.cuda() * gts.clone().cuda() + gt_mask.cuda()
            mask_aux_gts = cls_mask.cuda() * aux_gts.clone().cuda() + gt_mask.cuda()

            torch.save(mask_img.cpu(), patches_imgs_save_path + str(i) + '_' + str(cls_index[i]) + '_rgb.pt')

            torch.save(mask_gt.cpu(), patches_imgs_save_path + str(i) + '_' + str(cls_index[i]) + '_gt.pt')

            torch.save(mask_aux_gts.cpu(), patches_imgs_save_path + str(i) + '_' + str(cls_index[i]) + '_aux_gt.pt')

            cls_index[i] = cls_index[i] + 1

        if num_index.count(1) == len(num_index):
            break

        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info(
                    "collect class patches: {:d} / {:d} ".format(val_idx, len(train_csc_loader)))

    print(cls_index)
    np.savetxt(patches_nums_save_path + 'city_patches_num.txt', np.array(cls_index))

    return 1


def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    # if len(gpus)>1:
    for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    # else:
    #     for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #         #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    #         ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def tensor2img(tensor, heatmap=False, shape=(768, 768)):
    np_arr = tensor.detach().cpu().numpy()[0]
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    np_arr = (np_arr * 255).astype(np.uint8)
    # if np_arr.shape[0] == 1:
    #     np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    # print(np_arr.shape)
    np_arr = cv2.resize(np_arr, shape)
    # print(np_arr.shape)
    # np_arr = np_arr.transpose((2, 0, 1))
    np_arr = np.expand_dims(np_arr, axis=0)
    # print(np_arr.shape)
    # if heatmap:
    #     np_arr = cv2.resize(np_arr, shape)
    #     np_arr = cv2.applyColorMap(np_arr, cv2.COLORMAP_JET)
    return np_arr / 255


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())
    # print("backward_hook:", grad_in[0])
    # print("backward_hook:", grad_in[0].shape, grad_out[0].shape)


def farward_hook(module, input, output):
    fmap_block.append(output)
    # print("backward_hook:", output)
    # print("farward_hook:", input[0][0].shape, output[0].shape)


def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    train_loader, val_loaders, train_wild_loader, train_obj, extra_val_loaders, covstat_val_loaders = datasets.setup_loaders(args)

    _class_uniform_pct = args.class_uniform_pct
    args.class_uniform_pct = 0
    train_csc_loader, _, _, _, _, _ = datasets.setup_loaders(args)
    args.class_uniform_pct = _class_uniform_pct

    criterion, criterion_val, L1_loss = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

    teacher_model = network.get_net(args, criterion, criterion_aux)
    teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
    teacher_model = network.warp_network_in_dataparallel(teacher_model, args.local_rank)

    optim, scheduler = optimizer.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop

    # sample_cls_patch(train_csc_loader)

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        i = train(args, train_loader, train_wild_loader, net, teacher_model, optim, epoch, writer, scheduler, args.max_iter, criterion)

        train_loader.sampler.set_epoch(epoch + 1)
        train_wild_loader.sampler.set_epoch(epoch + 1)

        # update Mean teacher network
        if teacher_model is not None:
            alpha_teacher = 0.99
            teacher_model = update_ema_variables(ema_model=teacher_model, model=net, alpha_teacher=alpha_teacher, iteration=i)

        if len(extra_val_loaders) == 1:
            # Run validation only one time - To save models
            for dataset, val_loader in extra_val_loaders.items():
                validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=True)
        else:
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                              writer, epoch, "None", None, i, save_pth=True)

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()

        epoch += 1

    # Validation after epochs
    if len(val_loaders) == 1:
        # Run validation only one time - To save models
        for dataset, val_loader in val_loaders.items():
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
    else:
        if args.local_rank == 0:
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler, None, None, [],
                        writer, epoch, "None", None, i, save_pth=True)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)


def train(args, train_loader, wild_loader, net, teacher_model, optim, curr_epoch, writer, scheduler, max_iter, criterion):

    net.train()

    teacher_model.eval()

    cls_patches_len = list(np.loadtxt(patches_nums_save_path + 'city_patches_num.txt'))

    class_pixel_ratio = torch.tensor([509475373, 84222297, 315475529, 9076846, 12154035, 16964991, 2878741, 7636765,
                                      219931671, 16032511, 55381159, 16818673, 1863031, 96834205, 3699347, 3251826,
                                      3221320, 1364516, 5723359], dtype=torch.float)
    prob_class_pixel_ratio = torch.div(class_pixel_ratio, (torch.sum(class_pixel_ratio) + 1e-10))
    print(prob_class_pixel_ratio)

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    wild_loader_iter = enumerate(wild_loader)

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break

        inputs, gts, _, aux_gts = data

        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

        batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
            src_img, src_gt, src_aux_gt = ingredients
            src_img, src_gt, src_aux_gt = src_img.cuda(), src_gt.cuda(), src_aux_gt.cuda()
            start_ts = time.time()
            img_gt = None
            optim.zero_grad()
            ################### train stylized source domain #####################################
            _, inputs_wild = next(wild_loader_iter)
            input_wild = inputs_wild[0]
            input_wild = input_wild.cuda()
            x_mean = src_img.mean(dim=[2, 3], keepdim=True)  # B,C,1,1
            x_std = src_img.std(dim=[2, 3], keepdim=True) + 1e-7  # B,C,1,1
            x_mean, x_std = x_mean.detach(), x_std.detach()
            x_norm = (src_img - x_mean) / x_std
            new_mean = input_wild.mean(dim=[2, 3], keepdim=True)  # B,C,1,1
            new_std = input_wild.std(dim=[2, 3], keepdim=True) + 1e-7  # B,C,1,1
            sty_src_img = x_norm.cuda() * new_std.cuda() + new_mean.cuda()
            sty_src_gt = src_gt
            sty_src_aux_gt = src_aux_gt
            sty_src_outputs, sty_src_main_seg = net(sty_src_img, gts=sty_src_gt, aux_gts=sty_src_aux_gt, img_gt=img_gt,
                                            visualize=args.visualize_feature)
            sty_src_outputs_index = 0
            sty_src_main_loss = sty_src_outputs[sty_src_outputs_index]
            sty_src_outputs_index += 1
            sty_src_aux_loss = sty_src_outputs[sty_src_outputs_index]
            sty_src_outputs_index += 1
            sty_src_total_loss = sty_src_main_loss + (0.4 * sty_src_aux_loss)
            sty_src_log_total_loss = sty_src_total_loss.clone().detach_()
            torch.distributed.all_reduce(sty_src_log_total_loss, torch.distributed.ReduceOp.SUM)
            sty_src_log_total_loss = sty_src_log_total_loss / args.world_size
            train_total_loss.update(sty_src_log_total_loss.item(), batch_pixel_size)
            # src_total_loss.backward(retain_graph=True)
            sty_src_total_loss.backward()
            optim.step()
            sty_main_loss_value = sty_src_main_loss.item()
            sty_aux_loss_value = 0.4 * sty_src_aux_loss.item()
            time_meter.update(time.time() - start_ts)
            del sty_src_total_loss, sty_src_log_total_loss
            del src_total_loss, src_log_total_loss
            ################### train stylized source domain #####################################

            ################### train stylized sampling domain #####################################
            input_imgs = []
            input_gts = []
            input_aux_gts = []

            for bs_i in range(args.bs_mult):
                comps_imgs = src_img[bs_i].unsqueeze(0)
                comps_gts = src_gt[bs_i].unsqueeze(0)
                comps_aux_gts = src_aux_gt[bs_i].unsqueeze(0)

                # determine minority classes
                rare_class_ids = torch.nonzero(torch.lt(prob_class_pixel_ratio, balanced_ratio))
                # get minority classes id
                np_rare_class_ids = rare_class_ids.squeeze().numpy().tolist()
                # sample minority classes
                sample_index = min(2, len(np_rare_class_ids))
                # randomly select at least 2 minority classes
                sample_np_rare_class_ids = random.sample(np_rare_class_ids, sample_index)

                for cls_i in sample_np_rare_class_ids:
                    # randomly select a class patch
                    patch_ids = random.randint(0, int(cls_patches_len[cls_i]) - 1)

                    ############# get patches ###############################################
                    cls_img_patches = torch.load(
                        patches_imgs_save_path + str(cls_i) + '_' + str(patch_ids) + '_rgb.pt').cuda()
                    cls_gt_patches = torch.load(patches_imgs_save_path + str(cls_i) + '_' + str(patch_ids) + '_gt.pt').cuda()
                    cls_aux_gts_patches = torch.load(
                        patches_imgs_save_path + str(cls_i) + '_' + str(patch_ids) + '_aux_gt.pt').cuda()
                    ############# get patches ###############################################

                    ############# get CAM ###############################################
                    # # register hook
                    fh = teacher_model.module.layer4.register_forward_hook(farward_hook)
                    bh = teacher_model.module.layer4.register_backward_hook(backward_hook)
                    # save feature and grad list
                    preds = teacher_model(cls_img_patches, gts=cls_gt_patches, aux_gts=cls_aux_gts_patches, img_gt=img_gt, visualize=False)
                    cam_loss = criterion(preds, cls_gt_patches.long())
                    cam_loss.backward()
                    # uninstall hook
                    fh.remove()
                    bh.remove()
                    # get grad and features
                    layer1_grad = grad_block[-1]  # layer1_grad.shape [1, 64, 128, 128]
                    layer1_fmap = fmap_block[-1]
                    cam = layer1_grad[0, 0].mul(layer1_fmap[0][0, 0])
                    for l_i in range(1, layer1_grad.shape[1]):
                        cam += layer1_grad[0, l_i].mul(layer1_fmap[0][0, l_i])
                    # layer1_grad = layer1_grad.sum(1, keepdim=True)  # layer1_grad.shape [1, 1, 128, 128]
                    # layer1_fmap = layer1_fmap[0].sum(1, keepdim=True)  # 为了统一在tensor2img函数中调用
                    cam = cam.reshape((1, 1, *cam.shape))
                    cam_np = tensor2img(cam, heatmap=False, shape=(768, 768))
                    # print('cam_np: ', Counter(cam_np.flatten()))
                    cam_tensor = torch.from_numpy(cam_np).cuda(non_blocking=True)
                    # print('cam_tensor: ', Counter(cam_tensor.detach().cpu().numpy().flatten()))
                    grad_block.clear()
                    fmap_block.clear()
                    ############# get CAM ###############################################

                    ############# get img mask and gt mask ###############################################sss
                    if int(cls_i) == 0:
                        gt_mask = torch.ones(cls_gt_patches.shape).cuda(non_blocking=True)
                        gt_mask[cls_gt_patches == cls_i] = 0
                        gt_mask = 1 - gt_mask
                    else:
                        gt_mask = torch.zeros(cls_gt_patches.shape).cuda(non_blocking=True)
                        gt_mask[cls_gt_patches == cls_i] = 1
                    cls_img_patches = cls_img_patches * gt_mask
                    mask = torch.zeros(cls_img_patches.shape).cuda(non_blocking=True)
                    mask[cls_img_patches == 0] = 1
                    ############# get img mask and gt mask ###############################################

                    ############# get class-discriminative CAM mask ###############################################
                    # # print('cam_tensor: ', cam_tensor.shape)
                    # # print('gt2: ', gt_mask.shape)
                    cam_tensor_masked = cam_tensor * gt_mask
                    dis_cam = cam_tensor_masked * torch.gt(cam_tensor_masked, 0.9)
                    # print('dis_cam: ', Counter(dis_cam.detach().cpu().numpy().flatten()))
                    dis_mask = torch.ones(dis_cam.shape).cuda(non_blocking=True)
                    dis_mask[dis_mask == 0] = 0
                    ############# get class-discriminative CAM mask ###############################################

                    ############# update pixel ratio ###############################################
                    count_mask = torch.ones(cls_gt_patches.shape).cuda(non_blocking=True) * 255
                    count_mask[cls_gt_patches == cls_i] = 0
                    count_comps_gts = comps_gts * gt_mask + count_mask
                    counter = Counter(count_comps_gts.cpu().numpy().flatten())
                    for k, v in counter.items():
                        if int(k) in all_class_ids:
                            class_pixel_ratio[int(k)] = class_pixel_ratio[int(k)] - v
                    count_cls_gt_patches = cls_gt_patches * gt_mask + count_mask
                    counter = Counter(count_cls_gt_patches.cpu().numpy().flatten())
                    for k, v in counter.items():
                        if int(k) in all_class_ids:
                            class_pixel_ratio[int(k)] = class_pixel_ratio[int(k)] + v
                    ############# update pixel ratio ###############################################

                    ############# stylize patches ###############################################
                    sty_mask = torch.ones(cls_img_patches.shape).cuda(non_blocking=True)
                    sty_mask[cls_img_patches == 0] = 0
                    cls_norm = (cls_img_patches - cls_mean[cls_i]) / cls_std[cls_i]
                    new_mean = input_wild[bs_i].unsqueeze(0).mean(dim=[2, 3], keepdim=True)  # B,C,1,1
                    new_std = input_wild[bs_i].unsqueeze(0).std(dim=[2, 3], keepdim=True) + 1e-7  # B,C,1,1
                    cls_style = cls_norm.cuda() * new_std.cuda() + new_mean.cuda()
                    cls_style = cls_style * sty_mask
                    # cls_style = cls_style.cuda()
                    ############# stylize patches ###############################################

                    ############# preserve class-discriminative regions ###############################################
                    cls_style = cls_style.cuda() * (1 - dis_mask) + cls_img_patches * dis_mask
                    ############# preserve class-discriminative regions ###############################################

                    ############# paste patches onto template images ###############################################
                    comps_imgs = comps_imgs * mask
                    comps_imgs = comps_imgs + cls_style
                    comps_gts = comps_gts * (1 - gt_mask)
                    comps_gts = comps_gts + cls_gt_patches * gt_mask
                    comps_aux_gts = comps_aux_gts * (1 - gt_mask)
                    comps_aux_gts = comps_aux_gts + cls_aux_gts_patches * gt_mask
                    ############# paste patches onto template images ###############################################

                input_imgs.append(comps_imgs)
                input_gts.append(comps_gts)
                input_aux_gts.append(comps_aux_gts)

            sampling_input = torch.cat(input_imgs, dim=0).cuda()
            sampling_gt = torch.cat(input_gts, dim=0).long().cuda()
            sampling_aux_gt = torch.cat(input_aux_gts, dim=0).long().cuda()

            img_gt = None

            _, inputs_wild = next(wild_loader_iter)
            input_wild = inputs_wild[0]
            input_wild = input_wild.cuda()
            x_mean = sampling_input.mean(dim=[2, 3], keepdim=True)  # B,C,1,1
            x_std = sampling_input.std(dim=[2, 3], keepdim=True) + 1e-7  # B,C,1,1
            x_mean, x_std = x_mean.detach(), x_std.detach()
            x_norm = (sampling_input - x_mean) / x_std
            new_mean = input_wild.mean(dim=[2, 3], keepdim=True)  # B,C,1,1
            new_std = input_wild.std(dim=[2, 3], keepdim=True) + 1e-7  # B,C,1,1
            sty_sampling_img = x_norm.cuda() * new_std.cuda() + new_mean.cuda()
            sty_sampling_gt = sampling_gt
            sty_sampling_aux_gt = sampling_aux_gt
            sty_sampling_outputs, sty_sampling_main_seg = net(sty_sampling_img, gts=sty_sampling_gt, aux_gts=sty_sampling_aux_gt, img_gt=img_gt,
                                            visualize=args.visualize_feature)

            ############# semantic consistency constraint ###############################################
            gt_mask = torch.zeros(sty_sampling_gt.shape).cuda(non_blocking=True)
            gt_mask[src_gt == sty_sampling_gt] = 1
            mask = torch.ones(sty_sampling_gt.shape).cuda(non_blocking=True) * 255
            mask[src_gt == sty_sampling_gt] = 0
            sty_sampling_main_seg_sm = F.softmax(sty_sampling_main_seg, dim=1)
            mask_gt_prob = src_gt * gt_mask + mask
            mask_sty_sampling_prob = sty_sampling_main_seg_sm * gt_mask.unsqueeze(1) + mask.unsqueeze(1)
            # print('mask_gt_prob', mask_gt_prob.shape)
            # print('mask_sampling_prob', mask_sampling_prob.shape)
            sty_sampling_loss_mask_seg_consist = 0.5 * criterion(mask_sty_sampling_prob, mask_gt_prob.long())
            ############# semantic consistency constraint ###############################################

            ############# prediction consistency constraint ###############################################
            sampling_main_seg_sm = F.softmax(sampling_main_seg.detach(), dim=1).cpu().numpy() # sampling_main_seg.detach().cpu().data[0].numpy()
            sampling_main_seg_sm_label, sampling_main_seg_sm_prob = np.argmax(sampling_main_seg_sm, axis=1), np.max(sampling_main_seg_sm, axis=1)
            # # thres = []
            # # for i in range(19):
            # #     x = sampling_main_seg_sm_prob[sampling_main_seg_sm_label == i]
            # #     if len(x) == 0:
            # #         thres.append(0)
            # #         continue
            # #     x = np.sort(x)
            # #     thres.append(x[np.int(np.round(len(x) * 0.5))])
            # # # print(thres)
            # # thres = np.array(thres)
            # # thres[thres > 0.9] = 0.9
            # # for i in range(19):
            # #     sampling_main_seg_sm_label[(sampling_main_seg_sm_prob < thres[i]) * (sampling_main_seg_sm_label == i)] = 255
            sampling_main_seg_sm_ps_label = torch.from_numpy(np.asarray(sampling_main_seg_sm_label, dtype=np.uint8)).long().cuda()
            # loss_sty_consist = criterion(sty_sampling_main_seg, sampling_main_seg_sm_ps_label)
            loss_sty_consist = kl_categorical(sty_sampling_main_seg, sampling_main_seg_sm_ps_label)
            loss_pred_consist = loss_sty_consist + sty_sampling_loss_mask_seg_consist
            ############# prediction consistency constraint ###############################################

            sty_sampling_outputs_index = 0
            sty_sampling_main_loss = sty_sampling_outputs[sty_sampling_outputs_index]
            sty_sampling_outputs_index += 1
            sty_sampling_aux_loss = sty_sampling_outputs[sty_sampling_outputs_index]
            sty_sampling_outputs_index += 1
            sty_sampling_total_loss = sty_sampling_main_loss + (0.4 * sty_sampling_aux_loss) + loss_pred_consist
            sty_sampling_log_total_loss = sty_sampling_total_loss.clone().detach_()
            torch.distributed.all_reduce(sty_sampling_log_total_loss, torch.distributed.ReduceOp.SUM)
            sty_sampling_log_total_loss = sty_sampling_log_total_loss / args.world_size
            train_total_loss.update(sty_sampling_log_total_loss.item(), batch_pixel_size)
            sty_sampling_total_loss.backward()
            optim.step()
            sty_sampling_main_loss_value = sty_sampling_main_loss.item()
            sty_sampling_aux_loss_value = 0.4 * sty_sampling_aux_loss.item()
            sty_loss_consist_value = loss_sty_consist.item()
            sty_sampling_loss_mask_seg_consist_value = sty_sampling_loss_mask_seg_consist.item()
            time_meter.update(time.time() - start_ts)
            del sampling_total_loss, sampling_log_total_loss
            del sty_sampling_total_loss, sty_sampling_log_total_loss
            ################### train stylized sampling domain #####################################

            if args.local_rank == 0:
                # print('rank')
                if i % 50 == 49:
                    if args.visualize_feature:
                        visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')
                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [sampling_main_loss_value {:0.6f}], ' \
                          '[sampling_aux_loss_value {:0.6f}], [sampling_loss_mask_seg_consist_value {:0.6f}], ' \
                          '[sty_sampling_main_loss_value {:0.6f}], [sty_sampling_aux_loss_value {:0.6f}], ' \
                          '[sty_sampling_loss_mask_seg_consist_value {:0.6f}], ' \
                          '[sty_loss_consist_value {:0.6f}], [lr {:0.4f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg, sampling_main_loss_value,
                        sampling_aux_loss_value, sampling_loss_mask_seg_consist_value, sty_sampling_main_loss_value,
                        sty_sampling_aux_loss_value, sty_sampling_loss_mask_seg_consist_value, sty_loss_consist_value,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)

                    logging.info(msg)

                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter

        # break

    prob_class_pixel_ratio = torch.div(class_pixel_ratio, (torch.sum(class_pixel_ratio) + 1e-10))
    print(prob_class_pixel_ratio)

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.use_wtloss:
                output, f_cor_arr = net(inputs, visualize=True)
            else:
                output = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

        # if args.use_wtloss:
        #     visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

    return val_loss.avg

def validate_for_cov_stat(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    # net.train()#eval()
    net.eval()

    for val_idx, data in enumerate(val_loader):
        img_or, img_photometric, img_geometric, img_name = data   # img_geometric is not used.
        img_or, img_photometric = img_or.cuda(), img_photometric.cuda()
        # print('val')
        # print(img_photometric.shape)
        # print(img_or.shape)

        with torch.no_grad():
            net([img_photometric, img_or], cal_covstat=True)

        del img_or, img_photometric, img_geometric

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / 100", val_idx + 1)
        del data

        if val_idx >= 499:
            return


def visualize_matrix(writer, matrix_arr, iteration, title_str):
    stage = 'valid'

    for i in range(len(matrix_arr)):
        C = matrix_arr[i].shape[1]
        matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
        matrix = torch.clamp(torch.abs(matrix), max=1)
        matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                        torch.abs(matrix - 1.0)), 0)
        matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
        writer.add_image(stage + title_str + str(i), matrix, iteration)


def save_feature_numpy(feature_maps, iteration):
    file_fullpath = './visualization/feature_map/'
    file_name = str(args.date) + '_' + str(args.exp)
    B, C, H, W = feature_maps.shape
    for i in range(B):
        feature_map = feature_maps[i]
        feature_map = feature_map.data.cpu().numpy()   # H X D
        file_name_post = '_' + str(iteration * B + i)
        np.save(file_fullpath + file_name + file_name_post, feature_map)


if __name__ == '__main__':
    main()

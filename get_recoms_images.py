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

recoms_imgs_save_path = r'/mnt/CSCDG/CSC/coms_imgs/city/'
patches_imgs_save_path = r'/mnt/CSCDG/CSC/patches/city/'
patches_nums_save_path = r'/mnt/CSCDG/CSC/patches/'

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    # if args.csc:
    # _class_uniform_pct = args.class_uniform_pct
    # _bs_mult = args.bs_mult
    # args.class_uniform_pct = 0
    # args.bs_mult = 1
    # train_csc_loader, _, _, _, _ = datasets.setup_loaders(args)
    # args.bs_mult = 4
    train_recom_loader, _, _, _, _ = datasets.setup_loaders(args)
    # args.class_uniform_pct = _class_uniform_pct
    # args.bs_mult = _bs_mult

    conps_stuff_background_order = [0, 1, 2, 8, 9, 10]
    conps_stuff_foreground_order = [3, 4, 5, 6, 7]
    conps_ins_order = [11, 12, 13, 14, 15, 16, 17, 18]

    # conps_stuff_background_order = [0, 1, 2, 8, 10]
    # conps_stuff_foreground_order = [3, 4, 5, 6, 7]
    # conps_ins_order = [11, 12, 13, 15, 17, 18]

    criterion, criterion_val, L1_loss = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

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

    recom_iter = 0
    recom_epoch = 0
    while recom_iter < args.max_iter:
        for j, data in enumerate(train_recom_loader):
            if recom_iter >= args.max_iter:
                break
            inputs, gts, _, aux_gts = data
            # print(inputs.shape)
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

            for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
                src_img, src_gt, src_aux_gt = ingredients
                src_img, src_gt, src_aux_gt = src_img.cuda(), src_gt.cuda(), src_aux_gt.cuda()
                # print(src_img.shape)

                for bs_i in range(args.bs_mult):
                    random.shuffle(conps_stuff_background_order)
                    random.shuffle(conps_stuff_foreground_order)
                    random.shuffle(conps_ins_order)

                    img_composition(src_img[bs_i].unsqueeze(0), src_gt[bs_i].unsqueeze(0), src_aux_gt[bs_i].unsqueeze(0),
                                    conps_stuff_background_order,
                                    conps_stuff_foreground_order, conps_ins_order, recom_iter, bs_i)

            recom_iter = recom_iter + 1

            # del src_img, src_gt, src_aux_gt, inputs, gts, aux_gts

            if j % 20 == 0:
                if args.local_rank == 0:
                    logging.info(
                        "epoch {:d} \t iter: {:d} / {:d} \t curr_iter {:d}".format(recom_epoch, j, len(train_recom_loader), recom_iter))

        recom_epoch = recom_epoch + 1


    # while recom_iter < args.max_iter:
    #     imgs = torch.load(recoms_imgs_save_path + str(i) + '_rgb.pt').cuda()
    #
    #     # print(recoms_imgs_save_path + str(i) + '_' + str(bs_i) + '_gt.pt')
    #
    #     gts = torch.load(recoms_imgs_save_path + str(i) + '_gt.pt').long().cuda()
    #
    #     # print(recoms_imgs_save_path + str(i) + '_' + str(bs_i) + '_aux_gt.pt')
    #
    #     aux_gts = torch.load(recoms_imgs_save_path + str(i) + '_aux_gt.pt').long().cuda()
    #
    #     if recom_iter % 500 == 0:
    #         if args.local_rank == 0:
    #             logging.info(
    #                 "iter: {:d} / {:d}".format(recom_iter, len(train_recom_loader)))
    #
    #     recom_iter = recom_iter + 1


def img_composition(src_img, src_gt, src_aux_gt, conps_stuff_background_order, conps_stuff_foreground_order,
                    conps_ins_order, iter, bs_i):
    comps_imgs = src_img
    comps_gts = src_gt
    comps_aux_gts = src_aux_gt
    index = 0
    cls_patches_len = list(np.loadtxt(patches_nums_save_path + 'city_patches_num.txt'))
    for i in conps_stuff_background_order:
        ids = random.randint(0, int(cls_patches_len[i])-1)
        # comps_ids.append(ids)

        cls_img_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_rgb.pt').cuda()
        cls_gt_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_gt.pt').cuda()
        cls_aux_gts_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_aux_gt.pt').cuda()

        # print(cls_img_patches.shape) # torch.Size([1, 3, 768, 768])
        # print(cls_gt_patches.shape) # torch.Size([1, 768, 768])

        mask = torch.zeros(cls_img_patches.shape).cuda(non_blocking=True)
        mask[cls_img_patches == 0] = 1
        gt_mask = torch.zeros(cls_gt_patches.shape).cuda(non_blocking=True)
        gt_mask[cls_gt_patches == i] = 1
        comps_imgs = comps_imgs * mask
        comps_gts = comps_gts * (1 - gt_mask)
        comps_aux_gts = comps_aux_gts * (1 - gt_mask)
        comps_imgs = comps_imgs + cls_img_patches
        comps_gts = comps_gts + cls_gt_patches * gt_mask
        comps_aux_gts = comps_aux_gts + cls_aux_gts_patches * gt_mask
    for n in range(5):
        for i in conps_stuff_foreground_order:
            ids = random.randint(0, int(cls_patches_len[i])-1)
            # comps_ids.append(ids)

            cls_img_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_rgb.pt').cuda()
            cls_gt_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_gt.pt').cuda()
            cls_aux_gts_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_aux_gt.pt').cuda()

            mask = torch.zeros(cls_img_patches.shape).cuda(non_blocking=True)
            mask[cls_img_patches == 0] = 1
            gt_mask = torch.zeros(cls_gt_patches.shape).cuda(non_blocking=True)
            gt_mask[cls_gt_patches == i] = 1
            comps_imgs = comps_imgs * mask
            comps_gts = comps_gts * (1 - gt_mask)
            comps_aux_gts = comps_aux_gts * (1 - gt_mask)
            comps_imgs = comps_imgs + cls_img_patches
            comps_gts = comps_gts + cls_gt_patches * gt_mask
            comps_aux_gts = comps_aux_gts + cls_aux_gts_patches * gt_mask
    for n in range(5):
        for i in conps_ins_order:
            ids = random.randint(0, int(cls_patches_len[i])-1)
            # comps_ids.append(ids)

            cls_img_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_rgb.pt').cuda()
            cls_gt_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_gt.pt').cuda()
            cls_aux_gts_patches = torch.load(patches_imgs_save_path + str(i) + '_' + str(ids) + '_aux_gt.pt').cuda()

            mask = torch.zeros(cls_img_patches.shape).cuda(non_blocking=True)
            mask[cls_img_patches == 0] = 1
            gt_mask = torch.zeros(cls_gt_patches.shape).cuda(non_blocking=True)
            gt_mask[cls_gt_patches == i] = 1
            comps_imgs = comps_imgs * mask
            comps_gts = comps_gts * (1 - gt_mask)
            comps_aux_gts = comps_aux_gts * (1 - gt_mask)
            comps_imgs = comps_imgs + cls_img_patches
            comps_gts = comps_gts + cls_gt_patches * gt_mask
            comps_aux_gts = comps_aux_gts + cls_aux_gts_patches * gt_mask

    # # print(comps_imgs.shape) # torch.Size([1, 3, 768, 768])
    # save_comps_imgs = comps_imgs.squeeze(0).transpose(0, 1).transpose(1, 2).cpu().numpy()
    # # print(save_comps_imgs.shape)
    # save_comps_imgs = Image.fromarray(np.uint8(save_comps_imgs))
    # save_comps_imgs.save(recoms_imgs_save_path + str(iter) + '_' + str(index) + '_rgb.png')
    #
    # # print(comps_gts.shape)
    # save_comps_gts = comps_gts.transpose(0, 1).transpose(1, 2).repeat(1, 1, 3).cpu().numpy()
    # # print(save_comps_gts.shape)
    # save_comps_gts = Image.fromarray(np.uint8(save_comps_gts))
    # save_comps_gts.save(recoms_imgs_save_path + str(iter) + '_' + str(index) + '_gt.png')

    # torch.save(comps_imgs.cpu(), recoms_imgs_save_path + str(iter) + '_' + str(bs_i) + '_rgb.pt')
    #
    # torch.save(comps_gts.cpu(), recoms_imgs_save_path + str(iter) + '_' + str(bs_i) + '_gt.pt')
    #
    # torch.save(comps_aux_gts.cpu(), recoms_imgs_save_path + str(iter) + '_' + str(bs_i) + '_aux_gt.pt')

    torch.save(comps_imgs.cpu(), recoms_imgs_save_path + str(iter) + '_rgb.pt')

    torch.save(comps_gts.cpu(), recoms_imgs_save_path + str(iter) + '_gt.pt')

    torch.save(comps_aux_gts.cpu(), recoms_imgs_save_path + str(iter) + '_aux_gt.pt')

    # del comps_imgs, comps_gts, comps_aux_gts, cls_img_patches, cls_gt_patches, cls_aux_gts_patches

    return 1


if __name__ == '__main__':
    main()

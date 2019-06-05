###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import os
import numpy as np
from tqdm import tqdm
from skimage import io

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion

from datasets import get_edge_dataset, test_batchify_fn
from models import get_edge_model
from visualize import visualize_prediction

from option import Options
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

def test(args):
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    
    # dataset
    if args.eval: # set split='val' for validation set testing
        testset = get_edge_dataset(args.dataset, split='val', mode='testval',
                                           transform=input_transform, crop_size=args.crop_size)
    else: # set split='vis' for visulization
        testset = get_edge_dataset(args.dataset, split='vis', mode='vis',
                                           transform=input_transform, crop_size=args.crop_size)

    # output folder
    if args.eval:
        outdir_list_side5 = []
        outdir_list_fuse = []
        for i in range(testset.num_class):
            outdir_side5 = '%s/%s/%s_val/side5/class_%03d'%(args.dataset, args.model, args.checkname, i+1)
            if not os.path.exists(outdir_side5):
                os.makedirs(outdir_side5)
            outdir_list_side5.append(outdir_side5)

            outdir_fuse = '%s/%s/%s_val/fuse/class_%03d'%(args.dataset, args.model, args.checkname, i+1)
            if not os.path.exists(outdir_fuse):
                os.makedirs(outdir_fuse)
            outdir_list_fuse.append(outdir_fuse)

    else:
        outdir = '%s/%s/%s_vis'%(args.dataset, args.model, args.checkname)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    
    model = get_edge_model(args.model, dataset=args.dataset,
                           backbone=args.backbone,
                           norm_layer=BatchNorm2d,
                           crop_size=args.crop_size,
                           ) 
    
    # resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.cuda:
        model = DataParallelModel(model).cuda()
    print(model)

    model.eval()
    tbar = tqdm(test_data)

    if args.eval:
        for i, (images, im_paths, im_sizes) in enumerate(tbar):
            with torch.no_grad():
                images = [image.unsqueeze(0) for image in images]
                images = torch.cat(images, 0)
                outputs = model(images.float())

                num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
                if num_gpus == 1:
                    outputs = [outputs]

                # extract the side5 output and fuse output from outputs
                side5_list = []
                fuse_list = []
                for i in range(len(outputs)): #iterate for n (gpu counts)
                    im_size = tuple(im_sizes[i].numpy())
                    output = outputs[i]

                    side5 = output[0].squeeze_()
                    side5 = side5.sigmoid_().cpu().numpy()
                    side5 = side5[:,0:im_size[1],0:im_size[0]]

                    fuse = output[1].squeeze_()
                    fuse = fuse.sigmoid_().cpu().numpy()
                    fuse = fuse[:,0:im_size[1],0:im_size[0]]

                    side5_list.append(side5)
                    fuse_list.append(fuse)

                for predict, impath in zip(side5_list, im_paths):
                        for i in range(predict.shape[0]):
                            predict_c = predict[i]
                            path = os.path.join(outdir_list_side5[i], impath)
                            io.imsave(path, predict_c)

                for predict, impath in zip(fuse_list, im_paths):
                        for i in range(predict.shape[0]):
                            predict_c = predict[i]
                            path = os.path.join(outdir_list_fuse[i], impath)
                            io.imsave(path, predict_c)
    else:
        for i, (images, masks, im_paths, im_sizes) in enumerate(tbar):
            with torch.no_grad():
                images = [image.unsqueeze(0) for image in images]
                images = torch.cat(images, 0) 
                outputs = model(images.float())

                num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
                if num_gpus == 1:
                    outputs = [outputs]

                # extract the side5 output and fuse output from outputs
                side5_list = []
                fuse_list = []
                for i in range(len(outputs)): #iterate for n (gpu counts)
                    im_size = tuple(im_sizes[i].numpy())
                    output = outputs[i]

                    side5 = output[0].squeeze_()
                    side5 = side5.sigmoid_().cpu().numpy()
                    side5 = side5[:,0:im_size[1],0:im_size[0]]

                    fuse = output[1].squeeze_()
                    fuse = fuse.sigmoid_().cpu().numpy()
                    fuse = fuse[:,0:im_size[1],0:im_size[0]]

                    side5_list.append(side5)
                    fuse_list.append(fuse)

                # visualize ground truth
                for gt, impath in zip(masks, im_paths):
                    outname = os.path.splitext(impath)[0] + '_gt.png'
                    path = os.path.join(outdir, outname)
                    visualize_prediction(args.dataset, path, gt)
                
                # visualize side5 output
                for predict, impath in zip(side5_list, im_paths):
                    outname = os.path.splitext(impath)[0] + '_side5.png'
                    path = os.path.join(outdir, outname)
                    visualize_prediction(args.dataset, path, predict)

                # visualize fuse output
                for predict, impath in zip(fuse_list, im_paths):
                    outname = os.path.splitext(impath)[0] + '_fuse.png'
                    path = os.path.join(outdir, outname)
                    visualize_prediction(args.dataset, path, predict)

def eval_model(args):
    if args.resume_dir is None:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))

    if os.path.splitext(args.resume_dir)[1] == '.tar':
        args.resume = args.resume_dir
        assert os.path.exists(args.resume_dir)
        test(args)

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    eval_model(args)

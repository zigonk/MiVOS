"""
Generate fusion data for the DAVIS dataset.
"""

# from dataset.yv_test_dataset import YouTubeVOSTestDataset
import os
from os import path
from argparse import ArgumentParser

import torch
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2

from model.propagation.prop_net import PropagationNetwork
from dataset.davis_test_dataset import DAVISTestDataset
from dataset.bl_test_dataset import BLTestDataset
from dataset.custom_yv_test_dataset import YouTubeVOSTestDataset
from inference_core import InferenceCore
from model.fusion_net import FusionNet

from progressbar import progressbar


"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/propagation_model.pth')
parser.add_argument('--fusion_model', default='saves/fusion.pth')
parser.add_argument('--davis_root', default='../DAVIS/2017')
parser.add_argument('--bl_root', default='../BL30K')
parser.add_argument('--yv_im_path', default='')
parser.add_argument('--yv_mask_path', default='')
parser.add_argument('--yv_metadata', default='')
parser.add_argument('--dataset', help='DAVIS/BL')
parser.add_argument('--output')
parser.add_argument('--separation', default=None, type=int)
parser.add_argument('--range', default=None, type=int)
parser.add_argument('--mem_freq', default=None, type=int)
parser.add_argument('--start', default=None, type=int)
parser.add_argument('--end', default=None, type=int)
parser.add_argument('--yv_part', default='0', type=str)
args = parser.parse_args()

davis_path = args.davis_root
bl_path = args.bl_root
yv_im_path = args.yv_im_path
yv_mask_path = args.yv_mask_path
yv_metadata = args.yv_metadata
yv_part = args.yv_part
out_path = args.output
dataset_option = args.dataset

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(yv_mask_path+'/31e0beaf99/0/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
if dataset_option == 'DAVIS':
    test_dataset = DAVISTestDataset(davis_path+'/trainval', imset='2017/train.txt')
elif dataset_option == 'BL':
    test_dataset = BLTestDataset(bl_path, start=args.start, end=args.end)
elif dataset_option == 'YVOS':
    test_dataset = YouTubeVOSTestDataset(yv_im_path, yv_mask_path, yv_metadata, yv_part)
else:
    print('Use --dataset DAVIS or --dataset BL')
    raise NotImplementedError

# test_dataset = BLTestDataset(args.bl, start=args.start, end=args.end, subset=load_sub_bl())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

# Load our checkpoint
prop_saved = torch.load(args.model)
prop_model = PropagationNetwork().cuda().eval()
prop_model.load_state_dict(prop_saved)

fusion_saved = torch.load(args.fusion_model)
fusion_model = FusionNet().cuda().eval()
fusion_model.load_state_dict(fusion_saved)



# Start evaluation
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    rgb = data['rgb'].cuda()
    msk = data['gt'][0].cuda()
    info = data['info']
    total_t = rgb.shape[1]
    target_id = info['target_id'][0]
    processor = InferenceCore(prop_model, fusion_model, rgb, num_objects=1, mem_freq=5)
    # Make this directory
    this_out_path = path.join(out_path, info['name'][0], info['eid'][0])
    os.makedirs(this_out_path, exist_ok=True)
    # if (os.path.exists(os.path.join(this_out_path, '{}.png'.format(info['target_frame'][0])))):
        # continue
    if (target_id == 0):
        previous_mask = None
    # Push mask of target id into memory
    usable_keys = []
    if msk.shape[0] != 0  and (msk[0,target_id] > 0.5).sum() > 10*10:
        usable_keys.append(0)
    if len(usable_keys) != 0:
        this_msk = msk[usable_keys]
        processor.interact(this_msk[:, target_id], target_id, add_interact=False, prop_range=0)
    if previous_mask is not None and msk.shape[0] != 0:
        msk[:,0] = torch.from_numpy(test_dataset.All_to_onehot(previous_mask[np.newaxis,:], info['labels'][0].numpy())[0]).float()

    # Fused mask from two frames nearby
    output_mask = None
    msk.cuda()
    current_time = time.time()
    for frame in range(0, total_t, args.separation):
        if (frame == target_id):
            continue
        usable_keys = []
        for k in range(msk.shape[0]):
            if (msk[k,frame] > 0.5).sum() > 10*10:
                usable_keys.append(k)
        if len(usable_keys) == 0:
            continue
        if len(usable_keys) > 5:
            # Memory limit
            usable_keys = usable_keys[:5]

        k = len(usable_keys)
        this_msk = msk[usable_keys]


        # Propagate
        if dataset_option == 'DAVIS':
            left_limit = 0
            right_limit = total_t-1
        else:
            left_limit = max(0, frame-args.range)
            right_limit = min(total_t-1, frame+args.range)
        
        pred_range = range(left_limit, right_limit+1)
        out_probs = processor.interact(this_msk[:,frame], frame)
        for kidx, obj_id in enumerate(usable_keys):
            output_mask = out_probs[target_id] * 255
        del out_probs
    end_time = time.time()
    process_time = end_time - current_time
    print(process_time)
    if (msk.shape[0] == 0):
        output_mask = np.zeros((msk.shape[-2], msk.shape[-1])).astype(np.uint8)
    elif (output_mask is None):
        original_masks = ((msk[0] > 0.5) * 255).cpu().numpy().astype(np.uint8)
        output_mask = original_masks[target_id][0]
    imgE = Image.fromarray(output_mask)
    imgE = imgE.convert('L')
    imgE.save(os.path.join(this_out_path, '{}.png'.format(info['target_frame'][0])))
    previous_mask = output_mask


    torch.cuda.empty_cache()
# Partially taken from STM's dataloader

import os
from os import path

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import json

from dataset.range_transform import im_normalization

class YouTubeVOSTestDataset(Dataset):
    def __init__(self, image_path, mask_path, metadata, part='0'):
        self.image_dir = image_path
        self.mask_dir = mask_path
        self.metadata = json.load(open(metadata))

        self.dataset = []
        self.shape = {}
        self.frames = {}
        vid_list = {}
        vid_list['0'] = ['a9f23c9150', '6cc8bce61a', '03fe6115d4', 'a46012c642', 'c42fdedcdd', 'ee9415c553', '7daa6343e6', '4fe6619a47', '0e8a6b63bb', '65e0640a2a', '8939473ea7', 'b05faf54f7', '5d2020eff8', 'a00c3fa88e', '44e5d1a969', 'deed0ab4fc', 'b205d868e6', '48d2909d9e', 'c9ef04fe59', '1e20ceafae', '0f3f8b2b2f', 'b83923fd72', 'cb06f84b6e', '17cba76927', '35d5e5149d', '62bf7630b3', '0390fabe58', 'bf2d38aefe', '8b7b57b94d', '8d803e87f7', 'c16d9a4ade', '1a1dbe153e', 'd975e5f4a9', '226f1e10f7', '6cb5b08d93', '77df215672', '466734bc5c', '94fa9bd3b5', 'f2a45acf1c', 'ba8823f2d2', '06cd94d38d', 'b772ac822a', '246e38963b', 'b5514f75d8', '188cb4e03d', '3dd327ab4e', '8e2e5af6a8', '450bd2e238', '369919ef49', 'a4bce691c6', '64c6f2ed76', '0782a6df7e', '0062f687f1', 'c74fc37224', 'f7255a57d0', '4f5b3310e3', 'e027ebc228', '30fe0ed0ce', '6a75316e99', 'a2948d4116', '8273b59141', 'abae1ce57d', '621487be65', '45dc90f558', '9787f452bf', 'cdcfd9f93a', '4f6662e4e0', '853ca85618', '13ca7bbcfd', 'f143fede6f', '92fde455eb', '0b0c90e21a', '5460cc540a', '182dbfd6ba', '85968ae408', '541ccb0844', '43115c42b2', '65350fd60a', 'eb49ce8027', 'e11254d3b9', '20a93b4c54', 'a0fc95d8fc', '696e01387c', 'fef7e84268', '72d613f21a', '8c60938d92', '975be70866', '13c3cea202', '4ee0105885', '01c88b5b60', '33e8066265', '8dea7458de', 'c280d21988', 'fd8cf868b2', '35948a7fca', 'e10236eb37', 'a1251195e7', 'b2256e265c', '2b904b76c9', '1ab5f4bbc5', '47d01d34c8', 'd7a38bf258', '1a609fa7ee', '218ac81c2d', '9f16d17e42', 'fb104c286f', 'eb263ef128', '37b4ec2e1a', '0daaddc9da', 'cd69993923', '31d3a7d2ee', '60362df585', 'd7ff44ea97', '623d24ce2b', '6031809500', '54526e3c66', '0788b4033d', '3f4bacb16a', '06a5dfb511', '9f21474aca', '7a19a80b19', '9a38b8e463', '822c31928a', 'd1ac0d8b81', 'eea1a45e49', '9f429af409', '33c8dcbe09', '9da2156a73', '3be852ed44', '3674b2c70a', '547416bda1', '4037d8305d', '29c06df0f2', '1335b16cf9', 'b7b7e52e02', 'bc9ba8917e', 'dab44991de', '9fd2d2782b', 'f054e28786', 'b00ff71889', 'eeb18f9d47', '559a611d86', 'dea0160a12', '257f7fd5b8', 'dc197289ef', 'c2bbd6d121', 'f3678388a7', '332dabe378', '63883da4f5', 'b90f8c11db', 'dce363032d', '411774e9ff', '335fc10235', '7775043b5e', '3e03f623bb', '19cde15c4b', 'bf4cc89b18', '1a894a8f98', 'f7d7fb16d0', '61fca8cbf1', 'd69812339e', 'ab9a7583f1', 'e633eec195', '0a598e18a8', 'b3b92781d9', 'cd896a9bee', 'b7928ea5c0', '69c0f7494e', 'cc1a82ac2a', '39b7491321', '352ad66724', '749f1abdf9', '7f26b553ae', '0c04834d61', 'd1dd586cfd', '3b72dc1941', '39bce09d8d', 'cbea8f6bea', 'cc7c3138ff', 'd59c093632', '68dab8f80c', '1e0257109e', '4307020e0f', '4b783f1fc5', 'ebe7138e58', '1f390d22ea', '7a72130f21', 'aceb34fcbe', '9c0b55cae5', 'b58a97176b', '152fe4902a', 'a806e58451', '9ce299a510', '97b38cabcc', 'f39c805b54', '0620b43a31', '0723d7d4fe', '7741a0fbce', '7836afc0c2', 'a7462d6aaf', '34564d26d8', '31e0beaf99']
        vid_list['1'] = vid_list['0'][:70]
        vid_list['2'] = vid_list['0'][70:140]
        vid_list['3'] = vid_list['0'][140:]
        vid_list['test'] = ['a9f23c9150']
        # Pre-reading
        print('Init dataset loader')
        for vid in vid_list[part]:
            vid_frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = vid_frames

            
            video_info = self.metadata['videos'][vid]

            _mask = np.array(Image.open(path.join(self.mask_dir, vid, '0', video_info['frames'][0] + '.png')).convert("P"))
            self.shape[vid] = np.shape(_mask)
            for eid in video_info['expressions'].keys():
                for ind in range(len(video_info['frames'])):
                    frames = video_info['frames']
                    target_frame = frames[ind]
                    left_ref = frames[max(0, ind - 1)]
                    right_ref = frames[min(ind + 1, len(frames) - 1)]
                    info = {
                        'vid': vid,
                        'eid': eid,
                        'target_frame': target_frame,
                        'left_ref': left_ref,
                        'right_ref': right_ref,
                    }
                    self.dataset.append(info)
        print('Finish')
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    # From STM's code
    def To_onehot(self, mask, labels):
        M = np.zeros((len(labels), mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k, l in enumerate(labels):
            M[k] = (mask == l).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks, labels):
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n], labels)
        return Ms

    def __getitem__(self, idx):
        query_info = self.dataset[idx]
        info = {}
        video = query_info['vid']
        eid = query_info['eid']
        left_ref = query_info['left_ref']
        right_ref = query_info['right_ref']
        target_frame = query_info['target_frame']
        info['name'] = video
        info['num_objects'] = 0
        info['frames'] = self.frames[video] 
        info['size'] = self.shape[video] # Real sizes
        info['gt_obj'] = {} # Frames with labelled objects
        info['target_frame'] = target_frame
        info['eid'] = eid

        vid_im_path = path.join(self.image_dir, video)
        vid_gt_path = path.join(self.mask_dir, video, eid)

        frames = self.frames[video]

        images = []
        masks = []
        start_ind = frames.index(left_ref + '.jpg')
        end_ind = frames.index(right_ref + '.jpg')

        for i, f in enumerate(frames[start_ind:end_ind+1]):
            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))
            
            mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
            fid = f.replace('.jpg', '')
            if (fid == target_frame):
                info['target_id'] = i
            if path.exists(mask_file) and (fid in [left_ref, right_ref, target_frame]):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
            else:
                # Mask not exists -> nothing in it
                masks.append(np.zeros(self.shape[video]))
        
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        
        # Construct the forward and backward mapping table for labels
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(self.All_to_onehot(masks, labels)).float()

        # images = images.unsqueeze(0)
        masks = masks.unsqueeze(2)

        # Resize to 480p
        h, w = masks.shape[-2:]
        if h > w:
            new_size = (h*480//w, 480)
        else:
            new_size = (480, w*480//h)
        images = F.interpolate(images, size=new_size, mode='bicubic', align_corners=False)
        masks = F.interpolate(masks, size=(1, *new_size), mode='nearest')

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.dataset)
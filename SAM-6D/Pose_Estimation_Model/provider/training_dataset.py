
import os
import sys
import json
import cv2
import trimesh
import numpy as np
import h5py
import pandas as pd
import torch
import torchvision.transforms as transforms

import imgaug.augmenters as iaa
from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                Affine, PiecewiseAffine, ElasticTransformation, pillike, LinearContrast)  # noqa

from data_utils import (
    load_im,
    io_load_gt,
    io_load_masks,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
    get_random_rotation,
    get_bbox,
)

class Dataset():
    def __init__(self, cfg, num_img_per_epoch=-1):
        self.cfg = cfg

        #self.data_dir = cfg.data_dir
        self.data_dir='/content/drive/MyDrive/bpc_opencv_dataset/ipd/'
        self.num_img_per_epoch = num_img_per_epoch
        self.min_visib_px = cfg.min_px_count_visib
        self.min_visib_frac = cfg.min_visib_fract
        self.dilate_mask = cfg.dilate_mask
        self.rgb_mask_flag = cfg.rgb_mask_flag
        self.shift_range = cfg.shift_range
        self.img_size = cfg.img_size
        self.n_sample_observed_point = cfg.n_sample_observed_point
        self.n_sample_model_point = cfg.n_sample_model_point
        self.n_sample_template_point = cfg.n_sample_template_point
        

        self.data_paths = [
            os.path.join(self.data_dir, 'train_pbr')
        ]
        self.model_paths = [
            os.path.join(self.data_dir, 'models')
        ]
        self.templates_paths = [
            os.path.join(self.data_dir, 'templates')
        ]

        self.dataset_paths = []
        for f in self.data_paths:
            self.dataset_paths.append(os.path.join(self.data_dir, 'train_pbr', "000049")) # TODO, testing for only one scene

        self.length = len(self.dataset_paths)
        print('Total {} images .....'.format(self.length))


        with open(os.path.join(self.data_dir, 'models/models_info.json')) as fr:
            self.model_info = [json.load(fr)]

        # gdrnpp aug 
        aug_code = (
            "Sequential(["
            "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
            "Sometimes(0.4, GaussianBlur((0., 3.))),"
            "Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),"
            "Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),"
            "Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),"
            "Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),"
            "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
            "Sometimes(0.3, Invert(0.2, per_channel=True)),"
            "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
            "Sometimes(0.5, Multiply((0.6, 1.4))),"
            "Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),"
            "Sometimes(0.5, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),"
            "Sometimes(0.5, Grayscale(alpha=(0.0, 1.0))),"
            "], random_order=True)"
            # cosy+aae
        )
        self.color_augmentor = eval(aug_code)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])



    def __len__(self):
        return self.length if self.num_img_per_epoch == -1 else self.num_img_per_epoch

    def reset(self):
        if self.num_img_per_epoch == -1:
            self.num_img_per_epoch = self.length

        num_img = self.length
        if num_img <= self.num_img_per_epoch:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_idx = np.random.choice(num_img, self.num_img_per_epoch, replace=False)


    def __getitem__(self, index):
        while True:  # return valid data for train
            processed_data = self.read_data(self.img_idx[index])
            if processed_data is None:
                index = self._rand_another(index)
                continue
            return processed_data

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def read_data(self, index):
        print(f"Starting read_data with index {index}")
        
        path_head = self.dataset_paths[index]
        print(f"Path head: {path_head}")
        
        if not self._check_path(os.path.join(self.data_dir, path_head)):
            print("Path check failed")
            return None

        # # gt_info loading
        # print("Loading gt_info...")
        # gt_info = io_load_gt(open(os.path.join(self.data_dir, path_head+'/scene_gt_info_cam1.json'), 'rb'))
        # valid_idx = []
        # print(gt_info)
        # for k, item in enumerate(gt_info):
        #     print(f"Item {k}: {item}")

        #     if item['px_count_valid'] >= self.min_visib_px and item['visib_fract'] >= self.min_visib_frac:
        #         valid_idx.append(k)


        # print(f"Found {len(valid_idx)} valid instances")
        # if len(valid_idx) == 0:
        #     print("No valid instances found")
        #     return None
            
        # num_instance = len(valid_idx)
        # valid_idx = valid_idx[np.random.randint(0, num_instance)]
        # gt_info = gt_info[valid_idx]
        # ...existing code...
        # gt_info loading
        print("Loading gt_info...")
        gt_info = io_load_gt(open(os.path.join(self.data_dir, path_head+'/scene_gt_info_cam1.json'), 'rb'))
        valid_idx = []
        print(gt_info)

        # Flatten gt_info if it's a dict of lists (frame_id -> [instances])
        if isinstance(gt_info, dict):
            all_items = []
            for frame_id, instances in gt_info.items():
                for k, item in enumerate(instances):
                    print(f"Frame {frame_id} Item {k}: {item}")
                    if item['px_count_valid'] >= self.min_visib_px and item['visib_fract'] >= self.min_visib_frac:
                        all_items.append((frame_id, k))
            print(f"Found {len(all_items)} valid instances")
            if len(all_items) == 0:
                print("No valid instances found")
                return None
            # Pick a random valid instance
            frame_id, inst_idx = all_items[np.random.randint(0, len(all_items))]
            gt_info = gt_info[frame_id][inst_idx]
            valid_idx = inst_idx  # <--- Add this line!
        else:
            for k, item in enumerate(gt_info):
                print(f"Item {k}: {item}")
                if item['px_count_valid'] >= self.min_visib_px and item['visib_fract'] >= self.min_visib_frac:
                    valid_idx.append(k)
            print(f"Found {len(valid_idx)} valid instances")
            if len(valid_idx) == 0:
                print("No valid instances found")
                return None
            valid_idx = valid_idx[np.random.randint(0, len(valid_idx))]
            gt_info = gt_info[valid_idx]
# ...existing code...

        # gt loading
        print("Loading gt...")
        gt = io_load_gt(open(os.path.join(self.data_dir, path_head+'/scene_gt_cam1.json'), 'rb'))[valid_idx]
        
        print(f"Processing object ID: {gt['obj_id']}")
        obj_id = gt['obj_id']
        target_R = np.array(gt['cam_R_m2c']).reshape(3,3).astype(np.float32)
        target_t = np.array(gt['cam_t_m2c']).reshape(3).astype(np.float32) / 1000.0

        # camera loading
        print("Loading camera parameters...")
        camera = json.load(open(os.path.join(self.data_dir, path_head+'/scene_camera_cam1.json.json'), 'rb'))
        K = np.array(camera['cam_K']).reshape(3,3)

        # template loading
        print("Loading template...")
        tem1_rgb, tem1_choose, tem1_pts = self._get_template(obj_id, 0)
        if tem1_rgb is None:
            print("Template loading failed")
            return None

        # mask loading
        print("Loading and processing mask...")
        mask = io_load_masks(open(os.path.join(self.data_dir, path_head+'.mask_visib.json'), 'rb'))[valid_idx]
        if np.sum(mask) == 0:
            print("Empty mask")
            return None

        print("Processing depth...")
        depth = load_im(os.path.join(self.data_dir, path_head+'.depth.png')).astype(np.float32)
        depth = depth * camera['depth_scale'] / 1000.0
        
        print("Processing points...")
        pts = get_point_cloud_from_depth(depth, K, [y1, y2, x1, x2])
        print(f"Generated {pts.shape[0]} points")
        pts = pts.reshape(-1, 3)[choose, :]

        target_pts = (pts - target_t[None, :]) @ target_R
        tem_pts = np.concatenate([tem1_pts, tem2_pts], axis=0)
        radius = np.max(np.linalg.norm(tem_pts, axis=1))
        flag = np.linalg.norm(target_pts, axis=1) < radius * 1.2 # for outlier removal

        pts = pts[flag]
        choose = choose[flag]

        if len(choose) < 32:
            return None

        if len(choose) <= self.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        pts = pts[choose_idx]

        # rgb
        rgb = load_im(os.path.join(self.data_dir, path_head+'.rgb.jpg')).astype(np.uint8) # TODO
        rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
        if np.random.rand() < 0.8:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        # rotation aug
        rand_R = get_random_rotation()
        tem1_pts = tem1_pts @ rand_R
        tem2_pts = tem2_pts @ rand_R
        target_R = target_R @ rand_R

        # translation aug
        add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
        target_t = target_t + add_t[0]
        add_t = add_t + 0.001*np.random.randn(pts.shape[0], 3)
        pts = np.add(pts, add_t)

        ret_dict = {
            'pts': torch.FloatTensor(pts),
            'rgb': torch.FloatTensor(rgb),
            'rgb_choose': torch.IntTensor(rgb_choose).long(),
            'translation_label': torch.FloatTensor(target_t),
            'rotation_label': torch.FloatTensor(target_R),
            'tem1_rgb': torch.FloatTensor(tem1_rgb),
            'tem1_choose': torch.IntTensor(tem1_choose).long(),
            'tem1_pts': torch.FloatTensor(tem1_pts),
            'tem2_rgb': torch.FloatTensor(tem2_rgb),
            'tem2_choose': torch.IntTensor(tem2_choose).long(),
            'tem2_pts': torch.FloatTensor(tem2_pts),
            'K': torch.FloatTensor(K),
        }
        return ret_dict

    def _get_template(self, obj_id, tem_index=1):
        info = self.model_info[0][obj_id]
        assert info['obj_id'] == obj_id
        file_base = os.path.join(
            self.templates_paths[0],
            f'obj_{obj_id}'
        ) # TODO not sure

        rgb_path = os.path.join(file_base, 'rgb_'+str(tem_index)+'.png') # TODO
        xyz_path = os.path.join(file_base, 'xyz_'+str(tem_index)+'.npy') # TODO
        mask_path = os.path.join(file_base, 'mask_'+str(tem_index)+'.png') # TODO
        if not os.path.exists(rgb_path): # TODO
            return None, None, None

        # mask
        mask = load_im(mask_path).astype(np.uint8) == 255
        bbox = get_bbox(mask)
        y1,y2,x1,x2 = bbox
        mask = mask[y1:y2, x1:x2]

        # rgb
        rgb = load_im(rgb_path).astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]
        if np.random.rand() < 0.8:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))

        # xyz
        choose = mask.astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point, replace=False)
        choose = choose[choose_idx]

        xyz = np.load(xyz_path).astype(np.float32)[y1:y2, x1:x2, :]
        xyz = xyz.reshape((-1, 3))[choose, :] * 0.1
        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)
        print("Processing complete, returning data")

        return rgb, choose, xyz

    def _check_path(self, path_head):
        keys = [
            '.camera.json',
            '.depth.png',
            '.gt_info.json',
            '.gt.json',
            '.mask_visib.json',
            '.rgb.jpg'
        ]

        for k in keys:
            if not os.path.exists(path_head + k): # TODO
                return True
        return True

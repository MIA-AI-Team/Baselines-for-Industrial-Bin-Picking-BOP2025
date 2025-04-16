import gorilla
import os
import sys
from PIL import Image
import os.path as osp
import numpy as np
import random
import importlib
import json
import torch
import torchvision.transforms as transforms
import cv2
import trimesh
import pycocotools.mask as cocomask

from seg_interface import run_segmentation


# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'sam6d')
sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

# Import utility functions
from utils.data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from utils.draw_utils import draw_detections

# Transformations for RGB images
rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PoseEstimator:
    def __init__(self, config_path="config/base.yaml", model_name="pose_estimation_model", 
                 checkpoint_path="checkpoints/sam-6d-pem-base.pth", det_score_thresh=0.37):
        """
        Initialize the pose estimator with configuration and model parameters.
        
        Args:
            config_path: Path to the configuration file
            model_name: Name of the model to use
            checkpoint_path: Path to the model checkpoint
            det_score_thresh: Detection score threshold
        """
        self.cfg = self._init_config(config_path, model_name, det_score_thresh)
        self.model = self._load_model(model_name, checkpoint_path)
        
        # Set random seeds for reproducibility
        random.seed(self.cfg.rd_seed)
        torch.manual_seed(self.cfg.rd_seed)
    
    def _init_config(self, config_path, model_name, det_score_thresh):
        """Initialize configuration for the pose estimator."""
        exp_id = 0
        iter_val = 600000
        gpus = "0"
        
        exp_name = model_name + '_' + \
            osp.splitext(config_path.split("/")[-1])[0] + '_id' + str(exp_id)
        log_dir = osp.join("log", exp_name)

        cfg = gorilla.Config.fromfile(config_path)
        cfg.exp_name = exp_name
        cfg.gpus = gpus
        cfg.model_name = model_name
        cfg.log_dir = log_dir
        cfg.test_iter = iter_val
        cfg.det_score_thresh = float(det_score_thresh)
        
        gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
        return cfg
    
    def _load_model(self, model_name, checkpoint_path):
        """Load the pose estimation model."""
        print("=> Creating model...")
        MODEL = importlib.import_module(model_name)
        model = MODEL.Net(self.cfg.model)
        model = model.cuda()
        model.eval()
        
        # Load checkpoint
        checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoint_path)
        gorilla.solver.load_checkpoint(model=model, filename=checkpoint)
        return model
        
    def visualize(self, rgb, pred_rot, pred_trans, model_points, K, save_path):
        """Visualize the pose estimation results."""
        img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
        prediction = Image.open(save_path)
        
        # Concat side by side in PIL
        rgb_img = Image.fromarray(np.uint8(rgb))
        img = np.array(img)
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb_img, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        return concat
    
    def _get_template(self, path, tem_index=1):
        """Get template from the template directory."""
        rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
        mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
        xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

        rgb = load_im(rgb_path).astype(np.uint8)
        xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
        mask = load_im(mask_path).astype(np.uint8) == 255

        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
        if self.cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

        rgb = cv2.resize(rgb, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))

        choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.cfg.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.cfg.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.cfg.n_sample_template_point, replace=False)
        choose = choose[choose_idx]
        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.cfg.img_size)
        return rgb, rgb_choose, xyz
    
    def get_templates(self, template_dir):
        """Extract templates from the template directory."""
        print("=> Extracting templates...")
        n_template_view = self.cfg.n_template_view
        all_tem = []
        all_tem_choose = []
        all_tem_pts = []

        total_nView = 42
        for v in range(n_template_view):
            i = int(total_nView / n_template_view * v)
            tem, tem_choose, tem_pts = self._get_template(template_dir, i)
            all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
            all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
            all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
        
        with torch.no_grad():
            all_tem_pts, all_tem_feat = self.model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)
            
        return all_tem_pts, all_tem_feat
    
    def get_test_data(self, camera, detections, ply_obj_path):
        """Prepare test data from camera, detections and object model."""
        print("=> Loading input data...")
        dets = []
        for det in detections:
            if det['score'] > self.cfg.det_score_thresh:
                dets.append(det)

        CONST_DEPTH_SCALE = 0.1
        depth_scale = np.array([CONST_DEPTH_SCALE])

        # Use Camera class attributes - need to handle color vs rgb naming
        K = np.array(camera.intrinsics).reshape(3, 3)
        whole_image = camera.rgb
        if len(whole_image.shape) == 2:
            whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
            
        whole_depth = camera.depth * depth_scale / 1000.0
        whole_pts = get_point_cloud_from_depth(whole_depth, K)

        mesh = trimesh.load_mesh(ply_obj_path)
        model_points = mesh.sample(self.cfg.n_sample_model_point).astype(np.float32) / 1000.0
        radius = np.max(np.linalg.norm(model_points, axis=1))

        all_rgb = []
        all_cloud = []
        all_rgb_choose = []
        all_score = []
        all_dets = []
        
        for inst in dets:
            seg = inst['segmentation']
            score = inst['score']

            # mask
            h, w = seg['size']
            try:
                rle = cocomask.frPyObjects(seg, h, w)
            except:
                rle = seg
            mask = cocomask.decode(rle)
            mask = np.logical_and(mask > 0, whole_depth > 0)
            if np.sum(mask) > 32:
                bbox = get_bbox(mask)
                y1, y2, x1, x2 = bbox
            else:
                continue
            mask = mask[y1:y2, x1:x2]
            choose = mask.astype(np.float32).flatten().nonzero()[0]

            # pts
            cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
            center = np.mean(cloud, axis=0)
            tmp_cloud = cloud - center[None, :]
            flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
            if np.sum(flag) < 4:
                continue
            choose = choose[flag]
            cloud = cloud[flag]

            if len(choose) <= self.cfg.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), self.cfg.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.cfg.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            cloud = cloud[choose_idx]

            # rgb
            rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
            if self.cfg.rgb_mask_flag:
                rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
            rgb = cv2.resize(rgb, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = rgb_transform(np.array(rgb))
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.cfg.img_size)

            all_rgb.append(torch.FloatTensor(rgb))
            all_cloud.append(torch.FloatTensor(cloud))
            all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
            all_score.append(score)
            all_dets.append(inst)

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_cloud).cuda() if all_cloud else torch.FloatTensor().cuda()
        ret_dict['rgb'] = torch.stack(all_rgb).cuda() if all_rgb else torch.FloatTensor().cuda()
        ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda() if all_rgb_choose else torch.IntTensor().cuda()
        ret_dict['score'] = torch.FloatTensor(all_score).cuda() if all_score else torch.FloatTensor().cuda()

        if len(all_cloud) > 0:
            ninstance = ret_dict['pts'].size(0)
            ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
            ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        
        return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets
    
    def run_pose_estimation(self, camera, detections, ply_obj_path, template_dir, output_dir=None):
        """
        Main function to run pose estimation.
        
        Args:
            camera: Camera object with color, depth and intrinsics
            detections: Detection results from segmentation model
            ply_obj_path: Path to the object 3D model
            template_dir: Directory containing template images
            output_dir: Directory to save results
            
        Returns:
            detections: Updated detections with pose estimation results
            vis_img: Visualization of the pose estimation results
        """
        # Extract templates
        all_tem_pts, all_tem_feat = self.get_templates(template_dir)
        
        # Prepare test data
        input_data, img, whole_pts, model_points, detections = self.get_test_data(
            camera, detections, ply_obj_path
        )
        
        # No valid detections
        if len(input_data.get('pts', [])) == 0:
            print("No valid detections found.")
            return detections, None
        
        ninstance = input_data['pts'].size(0)
        
        # Run model inference
        print("=> Running model...")
        with torch.no_grad():
            input_data['dense_po'] = all_tem_pts.repeat(ninstance, 1, 1)
            input_data['dense_fo'] = all_tem_feat.repeat(ninstance, 1, 1)
            out = self.model(input_data)
        
        # Process results
        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] * input_data['score']
        else:
            pose_scores = input_data['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000
        
        # Update detections with pose estimation results
        for idx, det in enumerate(detections):
            detections[idx]['score'] = float(pose_scores[idx])
            detections[idx]['R'] = pred_rot[idx].tolist()
            detections[idx]['t'] = pred_trans[idx].tolist()
        
        # Save results if output directory is provided
        if output_dir:
            print("=> Saving results...")
            os.makedirs(f"{output_dir}/sam6d_results", exist_ok=True)
            
            with open(os.path.join(f"{output_dir}/sam6d_results", 'detection_pem.json'), "w") as f:
                json.dump(detections, f)
            
            # Visualize results
            print("=> Visualizing...")
            save_path = os.path.join(f"{output_dir}/sam6d_results", 'vis_pem.png')
            valid_masks = pose_scores <= pose_scores.max()
            K = input_data['K'].detach().cpu().numpy()[valid_masks]
            vis_img = self.visualize(img, pred_rot[valid_masks], pred_trans[valid_masks], model_points*1000, K, save_path)
            vis_img.save(save_path)
            return detections, vis_img
        
        return detections, None

# Main function to integrate both segmentation and pose estimation
def run_sam6d_pipeline(camera, template_dir, ply_obj_path, output_dir=None, 
                      segmentor_model="fastsam", stability_score_thresh=0.97, 
                      det_score_thresh=0.37):
    """
    Complete SAM-6D pipeline that runs segmentation followed by pose estimation.
    
    Args:
        camera: Camera object with color, depth and intrinsics
        template_dir: Directory containing template images
        ply_obj_path: Path to the object 3D model
        output_dir: Directory to save results (default: directory of ply_obj_path)
        segmentor_model: Segmentation model type ("sam" or "fastsam")
        stability_score_thresh: Stability score threshold for segmentation
        det_score_thresh: Detection score threshold for pose estimation
        
    Returns:
        detections: Final detections with pose estimation results
        vis_img: Visualization of the pose estimation results
    """
    # If output_dir is not provided, use the directory of ply_obj_path
    if output_dir is None:
        output_dir = os.path.dirname(ply_obj_path)
    
    # Ensure the output directory exists
    os.makedirs(f"{output_dir}/sam6d_results", exist_ok=True)
    
    # Step 1: Run instance segmentation
    
    print("=> Running instance segmentation...")
    detections = run_segmentation(
        camera, template_dir, ply_obj_path, 
        segmentor_model, stability_score_thresh
    )
    
    # Step 2: Initialize pose estimator
    print("=> Initializing pose estimator...")
    pose_estimator = PoseEstimator(det_score_thresh=det_score_thresh)
    
    # Step 3: Run pose estimation
    print("=> Running pose estimation...")
    detections, vis_img = pose_estimator.run_pose_estimation(
        camera, detections, ply_obj_path, template_dir, output_dir
    )
    
    return detections, vis_img
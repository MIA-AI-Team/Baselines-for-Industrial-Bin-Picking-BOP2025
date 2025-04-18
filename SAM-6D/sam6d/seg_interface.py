from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2


import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23


from typing import List, Optional, Union
from scipy.spatial.transform import Rotation
import numpy as np

import torch

class InstanceSegmentator:
    def __init__(
        self,
        stability_score_thresh: float = 0.97,
        segmentor_model: str = "fastsam",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.segmentor_model = segmentor_model
        self.stability_score_thresh = stability_score_thresh
        self.device = device
        
        with initialize(version_base=None, config_path="configs"):
           self.cfg = compose(config_name='run_inference.yaml')

        if segmentor_model == "sam":
            with initialize(version_base=None, config_path="configs/model"):
                self.cfg.model = compose(config_name='ISM_sam.yaml')
            self.cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        elif segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="configs/model"):
                self.cfg.model = compose(config_name='ISM_fastsam.yaml')
        else:
            raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

        logging.info("Initializing model")
        self.model = instantiate(self.cfg.model)

        
        self.model.descriptor_model.model = self.model.descriptor_model.model.to(device)
        self.model.descriptor_model.model.device = device

        # if there is predictor in the model, move it to device
        if hasattr(self.model.segmentor_model, "predictor"):
            self.model.segmentor_model.predictor.model = (
                self.model.segmentor_model.predictor.model.to(device)
            )
        else:
            pass
            # model.segmentor_model.model.setup_model(device=device, verbose=True)
        logging.info(f"Moving models to {device} done!")
        print(self.cfg)
        
        
    def predict(self, camera, template_dir, ply_obj_path, output_dir=None):
        logging.info("Initializing template")

        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
            mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        self.model.ref_data = {}
        self.model.ref_data["descriptors"] = self.model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        self.model.ref_data["appe_descriptors"] = self.model.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
        # run inference
        rgb = camera.rgb
        detections = self.model.segmentor_model.generate_masks(rgb)
        detections = Detections(detections)
        query_decriptors, query_appe_descriptors = self.model.descriptor_model.forward(rgb, detections)

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.model.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor= self.model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        # compute the geometric score
        batch = self._batch_input_data(camera)
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
        self.model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

        mesh = trimesh.load_mesh(ply_obj_path)
        model_points = mesh.sample(2048).astype(np.float32) / 1000.0
        self.model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(self.device)

        image_uv = self.model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        geometric_score, visible_ratio = self.model.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=self.model.visible_thred
            )

        # final score
        final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))   
        detections.to_numpy()
        if output_dir:
            save_path = f"{output_dir}/sam6d_results/detection_ism"
            os.makedirs(save_path, exist_ok=True)
            detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
            # Detections is the json format file data
            detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        return detections
    
    
    def _batch_input_data(self, camera):
        batch = {}

        CONST_DEPTH_SCALE = 0.1

        depth_scale = np.array([CONST_DEPTH_SCALE])

        batch["depth"] = torch.from_numpy(camera.depth).unsqueeze(0).to(self.device)
        batch["cam_intrinsic"] = torch.from_numpy(camera.intrinsics).unsqueeze(0).to(self.device)
        batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(self.device)
        return batch


# def run_segmentation(camera, template_dir, ply_obj_path, segmentor_model, output_dir=None, stability_score_thresh = 0.97):
    


    



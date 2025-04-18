import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
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
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, save_path="tmp.png"):
    # Create a copy of the original image.
    img = rgb.copy()
    
    # Convert image to grayscale and then back to RGB to standardize processing.
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Generate distinct colors for each detection.
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33  # Blending factor for the overlay.

    # Process each detection.
    for det_idx, det in enumerate(detections):
        # Decode RLE segmentation into a binary mask.
        mask = rle_to_mask(det["segmentation"])
        
        # Detect edges using the Canny edge detector and dilate to enhance visibility.
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        
        # Compute object size as the number of pixels in the mask.
        object_size = int(np.sum(mask))
        
        # Determine the color based on the category id.
        obj_id = det["category_id"]
        temp_id = obj_id - 1  # Adjust index to match color list indexing.
        r = int(255 * colors[temp_id][0])
        g = int(255 * colors[temp_id][1])
        b = int(255 * colors[temp_id][2])
        
        # Overlay the color onto the detection area using alpha blending.
        img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
        img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
        img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
        
        # Highlight the object boundaries in white.
        img[edge, :] = 255
        
        # Compute the top-left point to place the annotation by finding the minimum (x, y)
        # coordinates in the mask's pixel indices.
        ys, xs = np.where(mask)
        if len(ys) > 0 and len(xs) > 0:
            top_left = (int(xs.min()), int(ys.min()))
        else:
            # Default to a fixed location if the mask is empty.
            top_left = (10, 10)
        
        # Prepare the annotation text with score and size.
        text = f"Score: {det['score']:.2f}, Size: {object_size} px"
        
        # Annotate the image by drawing the text. The text is white with a thickness of 2.
        cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert the NumPy image to a PIL image, save it, and reload it for side-by-side display.
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # Create a new blank image and concatenate the original and annotated images side-by-side.
    img_np = np.array(img)
    concat = Image.new('RGB', (img_np.shape[1] + prediction.size[0], img_np.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img_np.shape[1], 0))
    
    return concat

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)['0']
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
      pass
        # model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    
    logging.info("Initializing template")
    cad_filename = os.path.basename(cad_path)
    object_name = os.path.splitext(cad_filename)[0]

    template_dir = os.path.join(output_dir, f'templates/{object_name}')
    print(template_dir)
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
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    
    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )

    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
    detections.to_numpy()
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    
    # Extract scene_id and image_id from rgb_path
    rgb_dir = os.path.dirname(rgb_path)
    scene_dir = os.path.dirname(os.path.dirname(rgb_dir))
    scene_id = os.path.basename(scene_dir)  
    image_id = os.path.splitext(os.path.basename(rgb_path))[0]  
    
    # Extract objs_id from cad_path
   
    objs_id = object_name 
    
   # Save visualization with the new filename format
    vis_save_path = f"{output_dir}/sam6d_results/vis_ism_{scene_id}_{image_id}_{objs_id}.png"
    vis_img = visualize(rgb, detections, vis_save_path)
    vis_img.save(vis_save_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
    run_inference(
        args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
    )
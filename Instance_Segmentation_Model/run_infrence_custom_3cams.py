import os
import sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import gc
import logging
import os.path as osp
import argparse
import glob
import json

# For Hydra-based configuration
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Visualization and image processing
import cv2
import imageio.v2 as imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation

# Torch transforms
import torchvision.transforms as T
from torchvision.utils import save_image

# BOP / local project imports
from rich.progress import Progress
import trimesh

# Local utilities (assuming they are in your project)
from utils.inout import load_json, save_json_bop23
from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from segment_anything.utils.amg import rle_to_mask

logging.basicConfig(level=logging.INFO)


inv_rgb_transform = T.Compose(
    [
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ]
)

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.0
    best_det = None
    for mask_idx, det in enumerate(detections):
        if det["score"] > best_score:
            best_score = det["score"]
            best_det = det

    if best_det is None:
        return rgb  # If no detections, just return the original image

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255 * colors[temp_id][0])
    g = int(255 * colors[temp_id][1])
    b = int(255 * colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]
    img[edge, :] = 255

    img_pil = Image.fromarray(np.uint8(img))
    img_pil.save(save_path)
    prediction = Image.open(save_path)

    # Side-by-side concatenation
    img_np = np.array(img_pil)
    concat = Image.new('RGB', (img_np.shape[1] + prediction.size[0], img_np.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img_np.shape[1], 0))
    return concat


def visualize_all(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33
    
    for det_id, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        temp_id = obj_id - 1

        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]
        img[edge, :] = 255

        img_pil = Image.fromarray(np.uint8(img))
        # Save intermediate
        vis_ism_path = os.path.join(save_path, f"vis_ism.png")
        img_pil.save(vis_ism_path)
        prediction = Image.open(vis_ism_path)
        
        # Side-by-side concatenation
        img_np = np.array(img_pil)
        concat = Image.new('RGB', (img_np.shape[1] + prediction.size[0], img_np.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img_np.shape[1], 0))
        concat.save(vis_ism_path)


def batch_input_data(depth_path, cam_path, device):
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)

    
    cam = cam_info[next(iter(cam_info))]
    cam_K = np.array(cam['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam['depth_scale'])

    batch = {}
    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


def save_segmentation_masks(detections, output_dir):
    """
    Saves each segmentation mask from the detections to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mask in enumerate(detections.masks):
        # Convert the mask to NumPy if it's a torch.Tensor
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        # Remove singleton dimensions
        mask_np = np.squeeze(mask_np)
        
        # Warn if shape is not 2D
        if mask_np.ndim != 2:
            print(f"Warning: Mask {i} has unexpected shape {mask_np.shape}. Verify dimensions.")

        # Normalize mask data
        if mask_np.dtype == bool:
            mask_np = mask_np.astype(np.uint8) * 255
        elif mask_np.dtype != np.uint8:
            mask_np = (mask_np * 255).astype(np.uint8)

        # Convert to a PIL Image
        try:
            mask_image = Image.fromarray(mask_np)
        except Exception as e:
            print(f"Error converting mask {i} to image: {e}")
            continue
        
        mask_path = os.path.join(output_dir, f"detection_mask_{i}.png")
        mask_image.save(mask_path)


# -----------------------------------------------------------------------------
# 1) Single‐View Inference
# -----------------------------------------------------------------------------
def run_inference_single_view(
    model,
    output_dir,
    input_dir,
    template_folder,
    cad_folder,
    rgb_path,
    depth_path,
    cam_path,
    obj_id,
    im_id
):
    """
    This function encapsulates your original single‐camera inference logic, but
    uses the camera-specific paths passed in as arguments.
    """
    device = next(model.descriptor_model.model.parameters()).device

    # -------------------------------------------------------------------------
    #  A) Prepare templates & reference data
    # -------------------------------------------------------------------------
    template_dir = os.path.join(template_folder, f"obj_{obj_id:06d}")
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []

    for idx in range(num_templates):
        image_path = os.path.join(template_dir, f"rgb_{idx}.png")
        mask_path = os.path.join(template_dir, f"mask_{idx}.png")
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            print(f"Warning: Missing template or mask for idx {idx}")
            continue
        
        image_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
        boxes.append(mask_pil.getbbox())

        image_tensor = torch.from_numpy(np.array(image_pil) / 255.0).float()
        mask_tensor = torch.from_numpy(np.array(mask_pil) / 255.0).float()
        # Apply mask
        image_tensor = image_tensor * mask_tensor[:, :, None]
        templates.append(image_tensor)
        masks.append(mask_tensor.unsqueeze(-1))
    
    if len(templates) == 0:
        print("No templates found or loaded incorrectly.")
        return
    
    templates = torch.stack(templates).permute(0, 3, 1, 2)  # [T, 3, H, W]
    masks = torch.stack(masks).permute(0, 3, 1, 2)          # [T, 1, H, W]
    boxes = torch.tensor(np.array(boxes))

    processing_config = OmegaConf.create({"image_size": 224})
    proposal_processor = CropResizePad(processing_config.image_size)

    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
        templates, token_name="x_norm_clstoken"
    ).unsqueeze(0).data  # shape [1, T, D]
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
        templates, masks_cropped[:, 0, :, :]
    ).unsqueeze(0).data  # shape [1, T, D]

    # -------------------------------------------------------------------------
    #  B) Segment the single‐camera image
    # -------------------------------------------------------------------------
    rgb_pil = Image.open(rgb_path).convert("RGB")
    rgb_np = np.array(rgb_pil)
    detections_raw = model.segmentor_model.generate_masks(rgb_np)
    detections = Detections(detections_raw)

    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(rgb_np, detections)

    # Save segmentation masks
    mask_output_dir = os.path.join(output_dir, "segmentation_masks")
    save_segmentation_masks(detections, mask_output_dir)

    # -------------------------------------------------------------------------
    #  C) Matching descriptors (semantic, appearance, geometry)
    # -------------------------------------------------------------------------
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # Appearance
    appe_scores, ref_aux_descriptor = model.compute_appearance_score(
        best_template, pred_idx_objects, query_appe_descriptors
    )

    # Geometry
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

    cad_path = os.path.join(cad_folder, f"obj_{obj_id:06d}.ply")
    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)

    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)
    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
    )

    # Final score
    final_score = (semantic_score + appe_scores + geometric_score * visible_ratio) / (1 + 1 + visible_ratio)
    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))

    # Save results
    detections.to_numpy()
    results_dir = os.path.join(output_dir, "sam6d_results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "detection_ism")

    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections_json = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
    save_json_bop23(save_path + ".json", detections_json)

    # Visualization
    visualize_all(rgb_pil, detections, results_dir)

    # Cleanup GPU memory
    torch.cuda.empty_cache()
    gc.collect()


# -----------------------------------------------------------------------------
# 2) Run Inference for One Camera
# -----------------------------------------------------------------------------
def run_inference_for_one_camera(
    model,
    output_dir,
    input_dir,
    template_folder,
    cad_folder,
    camera_id,
    im_id,
    obj_id
):
    """
    Builds the file paths for the given camera_id (e.g., 1, 2, 3)
    and calls run_inference_single_view.
    """
    # Example folder structure for camera:
    #   rgb_cam1/<im_id>.png
    #   depth_cam1/<im_id>.png
    #   scene_camera_cam1.json
    # Adjust if your naming pattern differs.
    rgb_path = os.path.join(input_dir, f"rgb_cam{camera_id}", f"{im_id:06d}.png")
    depth_path = os.path.join(input_dir, f"depth_cam{camera_id}", f"{im_id:06d}.png")

    # The camera info JSON is typically named similarly. We'll just glob for it:
    cam_file_pattern = os.path.join(input_dir, f"scene_camera_cam{camera_id}.json")
    cam_files = glob.glob(cam_file_pattern)
    if len(cam_files) == 0:
        raise FileNotFoundError(f"No camera info found for cam {camera_id} in {input_dir}")
    cam_path = cam_files[0]

    # Now call the single‐view inference
    run_inference_single_view(
        model=model,
        output_dir=output_dir,
        input_dir=input_dir,
        template_folder=template_folder,
        cad_folder=cad_folder,
        rgb_path=rgb_path,
        depth_path=depth_path,
        cam_path=cam_path,
        obj_id=obj_id,
        im_id=im_id
    )


# -----------------------------------------------------------------------------
# 3) Run Inference with 3 Cameras
# -----------------------------------------------------------------------------
def run_inference_with_3_cameras(model, output_dir, input_dir, template_folder, cad_folder):
    """
    For a single scene/folder (input_dir), we run inference for 3 cameras:
    rgb_cam1, rgb_cam2, rgb_cam3 (and depth_cam1, depth_cam2, depth_cam3).
    """

    logging.info("Initializing templates / reading scene info ...")

    # This JSON determines im_id and obj_id
    test_targets_path = '/content/drive/MyDrive/bpc_opencv_dataset/ipd/test_targets_bop19.json'
    with open(test_targets_path, "r") as f:
        test_target = json.load(f)

    # Identify the scene_id from the input_dir name
    normalized_input_dir = os.path.normpath(input_dir)
    basename = os.path.basename(normalized_input_dir)

    if not basename.isdigit():
        basename = os.path.basename(os.path.dirname(normalized_input_dir))
        if not basename.isdigit():
            raise ValueError(f"Could not find a numeric scene id in: {input_dir}")

    scene_id = int(basename)
    # Find corresponding im_id and obj_id
    im_id = [item['im_id'] for item in test_target if item['scene_id'] == scene_id][0]
    obj_id = [item['obj_id'] for item in test_target if item['scene_id'] == scene_id][0]

    # Run inference for each camera
    for cam_number in [1, 2, 3]:
        logging.info(f"Running inference for scene {scene_id}, camera {cam_number}")
        cam_output_dir = os.path.join(output_dir, f"cam{cam_number}")
        os.makedirs(cam_output_dir, exist_ok=True)

        run_inference_for_one_camera(
            model=model,
            output_dir=cam_output_dir,
            input_dir=input_dir,
            template_folder=template_folder,
            cad_folder=cad_folder,
            camera_id=cam_number,
            im_id=im_id,
            obj_id=obj_id
        )

        # Cleanup GPU memory (extra safety)
        torch.cuda.empty_cache()
        gc.collect()


# -----------------------------------------------------------------------------
# 4) Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM (sam or fastsam)")
    parser.add_argument("--output_dir", help="Path to root directory for output")
    parser.add_argument("--input_dir", help="Path to input data (scenes)")
    parser.add_argument("--template_dir", help="Path to templates")
    parser.add_argument("--cad_dir", help="Path to CAD files (mm)")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Hydra-based config initialization
    # -------------------------------------------------------------------------
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    # Choose model (SAM or fastSAM)
    if args.segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = args.stability_score_thresh
    elif args.segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError(f"The segmentor_model {args.segmentor_model} is not supported!")

    logging.info("Initializing model ...")
    model = instantiate(cfg.model)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device

    # If there's a predictor, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = model.segmentor_model.predictor.model.to(device)
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)

    logging.info(f"Moving models to {device} done!")

    # -------------------------------------------------------------------------
    # Process Scenes
    # -------------------------------------------------------------------------
    input_folders = sorted(os.listdir(args.input_dir))

    with Progress() as progress:
        input_tqdm = progress.add_task('Processing Scenes', total=len(input_folders))

        for input_folder in input_folders:
            input_dir = os.path.join(args.input_dir, input_folder)
            output_dir = os.path.join(args.output_dir, input_folder)
            os.makedirs(os.path.join(output_dir, "sam6d_results"), exist_ok=True)

            # Run inference for 3 cameras in this scene
            run_inference_with_3_cameras(
                model=model,
                output_dir=output_dir,
                input_dir=input_dir,
                template_folder=args.template_dir,
                cad_folder=args.cad_dir
            )

            # Cleanup memory again if needed
            torch.cuda.empty_cache()
            gc.collect()

            progress.update(input_tqdm, advance=1)

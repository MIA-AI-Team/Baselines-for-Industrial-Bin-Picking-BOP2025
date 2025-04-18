import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation # Still needed for calc_pose_matrix if R,t loaded
import sys
from typing import List, Optional, Dict, Any
import glob
import json
import re # For extracting camera name from filename
import matplotlib.pyplot as plt
from pose_interface import run_sam6d_pipeline
import argparse
# --- Helper Functions ---

def calc_pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculates the 4x4 pose matrix (world-to-camera) from R and t."""
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix R must be 3x3, but got shape {R.shape}")
    if t.shape != (3,) and t.shape != (3, 1) and t.shape != (1, 3):
         raise ValueError(f"Translation vector t must be 3x1, 1x3 or (3,), but got shape {t.shape}")

    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R
    pose_matrix[:3, 3] = t.flatten()
    return pose_matrix

def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
    r = Rotation.from_matrix(rot)
    q = r.as_quat()
    return q


class Camera:
    """
    Represents a camera with its pose, intrinsics, and image data,
    initialized directly from data arrays. Simplified for verification.
    """
    def __init__(
        self,
        name: str,
        pose: np.ndarray, # Expecting 4x4 world-to-camera matrix
        intrinsics: np.ndarray, # Expecting 3x3 K matrix
        rgb: np.ndarray, # Expecting HxWxC BGR image from cv2.imread
        depth: Optional[np.ndarray] = None, # Optional depth
    ):
        if pose.shape != (4, 4):
            raise ValueError(f"Pose matrix for camera {name} must be 4x4, got {pose.shape}")
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Intrinsics matrix for camera {name} must be 3x3, got {intrinsics.shape}")
        if rgb is not None and (rgb.ndim != 3 or rgb.shape[2] != 3):
             if rgb.ndim == 2:
                 print(f"Warning: Grayscale image provided for camera {name}. Tiling to 3 channels.")
                 rgb = np.tile(rgb[:,:,None], (1, 1, 3))
             else:
                raise ValueError(f"RGB image for camera {name} must be HxWxC (3 channels), got {rgb.shape}")

        self.name: str = name
        self.pose: np.ndarray = pose
        self.intrinsics: np.ndarray = intrinsics
        self.rgb: Optional[np.ndarray] = rgb # Allow None if image loading fails
        self.depth: Optional[np.ndarray] = depth

class StandalonePoseEstimator:
    def __init__(self):
        print("Initializing StandalonePoseEstimator")

    def get_pose_estimates(
        self,
        object_ids: List[int],
        cam_1: Camera,
        cam_2: Camera,
        cam_3: Camera,
        photoneo: Optional[Camera] = None,
    ) -> List[Dict[str, Any]]:
        """
        Estimates poses for given object IDs using data from cam_1, cam_2, cam_3.
        """
        pose_estimates_results = []
        # Explicit check for the required cameras passed to this function
        if not all([cam_1, cam_2, cam_3]):
             missing = []
             if not cam_1: missing.append("cam_1")
             if not cam_2: missing.append("cam_2")
             if not cam_3: missing.append("cam_3")
             print(f"Error: get_pose_estimates requires valid Camera objects for {missing}.")
             # Depending on desired behavior, could raise error or return empty
             # Raising error is safer if these are truly required.
             raise ValueError(f"Missing required Camera objects for pose estimation: {missing}")

        cams = [cam_1, cam_2, cam_3] # Use only the three required cams for BPC Capture

        for object_id in object_ids:
            print(f"Running SAM-6D pipeline for object_id: {object_id}")
            # template dir is dataset_dir/templates/obj_000000 where 000000 is the object_id
            template_dir = os.path.join(DATASET_DIR, "templates", f"obj_{object_id:06d}")
            ply_obj_path = os.path.join(DATASET_DIR, "models", f"obj_{object_id:06d}.ply")
            output_dir = os.path.join(DATASET_DIR, "results", f"obj_{object_id:06d}")
            if not os.path.exists(template_dir):
                print(f"Warning: Template directory does not exist: {template_dir}")
                continue
            if not os.path.exists(ply_obj_path):
                print(f"Warning: Object model file does not exist: {ply_obj_path}")
                continue
            
            run_sam6d_pipeline(
                camera=cams[0],  # Using only cam_1
                template_dir=template_dir,
                ply_obj_path=ply_obj_path,
                output_dir=output_dir,
                segmentor_model="fastsam",
                stability_score_thresh=0.97,
                det_score_thresh=0.37
            )

            # images = [cam.rgb for cam in cams]
            # RTs = [cam.pose for cam in cams]
            # Ks = [cam.intrinsics for cam in cams]

            # print(f"Creating Capture object for object_id: {object_id}")
            # capture = Capture(images, Ks, RTs, object_id)

            # print("Running detection...")
            # t_start = time.time()
            # try:
            #     # Using private methods - replace with public API if available
            #     detections = pose_estimator._detect(capture)
            #     print(f"Detection found {len(detections)} potential objects.")
            #     pose_predictions = pose_estimator._match(capture, detections)
            #     print(f"Matching resulted in {len(pose_predictions)} predictions.")
            #     pose_estimator._estimate_rotation(pose_predictions)
            #     print(f"Pose estimation completed in {time.time() - t_start:.3f} seconds.")

            #     for detection in pose_predictions:
            #         if hasattr(detection, 'pose') and isinstance(detection.pose, np.ndarray) and detection.pose.shape == (4,4):
            #             estimate = {
            #                 "obj_id": object_id,
            #                 "score": getattr(detection, 'score', 1.0),
            #                 "pose": detection.pose
            #             }
            #             pose_estimates_results.append(estimate)
            #             print(f"  Added estimate for obj {object_id} with score {estimate['score']:.3f}")
            #         else:
            #              print(f"  Warning: Skipping detection for obj {object_id} due to missing/invalid pose attribute.")

            # except Exception as e:
            #     print(f"Error during pose estimation pipeline for object {object_id}: {e}")
            #     import traceback
            #     traceback.print_exc()

        return pose_estimates_results



def load_camera_params(scene_dir: str) -> Dict[str, Dict[str, List[Optional[np.ndarray]]]]:
    """Loads camera parameters (K, R, t) for all images by discovering scene_camera_*.json files."""
    restructured_params = {}
    max_image_index = -1
    param_files = glob.glob(os.path.join(scene_dir, 'scene_camera_*.json'))
    if not param_files: return {}
    print(f"Found camera parameter files: {param_files}")
    cam_name_pattern = re.compile(r'scene_camera_([a-zA-Z0-9_]+)\.json$')
    for param_file in param_files:
        match = cam_name_pattern.search(os.path.basename(param_file))
        if not match: continue
        cam_name = match.group(1)
        print(f"  Processing parameters for discovered camera '{cam_name}' from: {param_file}")
        try:
            with open(param_file, 'r') as f: cam_data = json.load(f)
        except Exception as e:
             print(f"  Warning: Could not read/parse {param_file}. Skipping. Error: {e}")
             restructured_params[cam_name] = {'K': [], 'R': [], 't': []}; continue
        image_indices = sorted([int(k) for k in cam_data.keys() if k.isdigit()])
        if not image_indices:
            print(f"  Warning: No valid image indices in {param_file}. Skipping.")
            restructured_params[cam_name] = {'K': [], 'R': [], 't': []}; continue
        current_max_idx = max(image_indices)
        if current_max_idx > max_image_index: max_image_index = current_max_idx
        num_images_for_cam = current_max_idx + 1
        K_list, R_list, t_list = [None]*num_images_for_cam, [None]*num_images_for_cam, [None]*num_images_for_cam
        for img_idx in image_indices:
            img_idx_str = str(img_idx)
            if img_idx_str in cam_data:
                img_data = cam_data[img_idx_str]
                try:
                    K = np.array(img_data['cam_K'], dtype=np.float64).reshape(3, 3)
                    R = np.array(img_data['cam_R_w2c'], dtype=np.float64).reshape(3, 3)
                    t = np.array(img_data['cam_t_w2c'], dtype=np.float64).reshape(3, 1)
                    K_list[img_idx], R_list[img_idx], t_list[img_idx] = K, R, t.flatten()
                except (KeyError, ValueError) as e: print(f"  Warn: Param error img {img_idx_str} in {param_file}. None. Err: {e}")
        restructured_params[cam_name] = {'K': K_list, 'R': R_list, 't': t_list}
        print(f"  Processed '{cam_name}' up to index {current_max_idx}.")
    if max_image_index >= 0:
        num_images_total = max_image_index + 1
        print(f"\nNormalizing param lists to {num_images_total}.")
        for cam_name in restructured_params.keys():
            for key in ['K', 'R', 't']:
                current_list = restructured_params[cam_name][key]
                current_len = len(current_list)
                if current_len < num_images_total: current_list.extend([None]*(num_images_total - current_len))
    return restructured_params

def load_images(scene_dir: str, cam_names: List[str], image_id_str: str, img_folder_prefix="rgb") -> Dict[str, Optional[np.ndarray]]:
    """Loads RGB images for the given camera names and image ID. Returns None if loading fails."""
    images = {}
    print(f"\nLoading images for image ID: {image_id_str}")
    for cam_name in cam_names:
        folder_name = f"{img_folder_prefix}_{cam_name}"
        search_pattern = os.path.join(scene_dir, folder_name, f"{image_id_str}.*")
        image_paths = glob.glob(search_pattern)
        img = None # Default to None
        if not image_paths:
             for ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                  fpath = os.path.join(scene_dir, folder_name, f"{image_id_str}.{ext}")
                  if os.path.exists(fpath): image_paths.append(fpath); break
        if image_paths:
            image_path = image_paths[0]
            try:
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is None: print(f"  Warning: cv2.imread failed for: {image_path}")
                else: print(f"  Loaded image for '{cam_name}': {image_path} (shape: {img.shape})")
            except Exception as e: print(f"  Warning: Error reading {image_path}: {e}")
        else: print(f"  Warning: Could not find image for '{cam_name}' using pattern: {search_pattern}.")
        images[cam_name] = img # Store image array or None
    return images

if __name__ == "__main__":
    # --- Argument Parser ---

    parser = argparse.ArgumentParser(description="Run pose estimation on a dataset.")
    parser.add_argument('--dataset_dir', type=str, default="/content/drive/MyDrive/bpc_opencv_dataset/ipd",
                        help="Path to the dataset directory")
    parser.add_argument('--scene_id', type=str, default="000008",
                        help="Scene ID (e.g., 000008)")
    parser.add_argument('--image_id', type=int, default=0,
                        help="Image ID (e.g., 0)")
    parser.add_argument('--object_ids', type=int, nargs='+', default=[14],
                        help="List of object IDs to test (e.g., 14)")

    args = parser.parse_args()

    # --- Set Variables from Arguments ---
    DATASET_DIR = args.dataset_dir
    MODEL_DIR = os.path.join(DATASET_DIR, "models")
    SCENE_ID = args.scene_id
    IMAGE_ID = args.image_id
    OBJECT_IDS_TO_TEST = args.object_ids

    # --- Step 3: Prepare Paths ---
    scene_dir = os.path.join(DATASET_DIR, "test", SCENE_ID)
    image_id_str = f"{IMAGE_ID:06d}"

    print("--- Starting Data Loading Verification via get_pose_estimates ---")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Scene Directory: {scene_dir}")
    print(f"Image Index: {IMAGE_ID} (Filename ID: {image_id_str})")
    print(f"Object IDs to test: {OBJECT_IDS_TO_TEST}")

    if not os.path.isdir(scene_dir):
        print(f"\nError: Scene directory not found: {scene_dir}")
    else:
        # try:
        # --- Step 4: Discover cameras and load parameters ---
        print("\nDiscovering cameras and loading parameters...")
        all_cam_params = load_camera_params(scene_dir)
        discovered_cam_names = sorted(list(all_cam_params.keys()))
        if not discovered_cam_names: raise FileNotFoundError(f"No camera param files found in {scene_dir}.")
        print(f"Discovered cameras: {discovered_cam_names}")

        # --- Step 5: Load Images for the SPECIFIC image_id ---
        rgb_images = load_images(scene_dir, discovered_cam_names, image_id_str, "rgb")
        depth_images = load_images(scene_dir, discovered_cam_names, image_id_str, "depth")

        # --- Step 6: Create Camera Objects ---
        print("\nCreating Camera objects...")
        cameras = {}
        for cam_name in discovered_cam_names:
            print(f"  Processing camera: {cam_name}")
            rgb_img = rgb_images.get(cam_name) # Get image array or None
            depth_img = depth_images.get(cam_name) # Get depth array or None
            if cam_name not in all_cam_params or not all_cam_params[cam_name]['K']:
                print(f"    - Parameters not loaded. Skipping object creation.")
                continue
            # Allow Camera object creation even if image failed to load (rgb_img is None)
            # But parameters must exist for this IMAGE_ID

            try:
                if IMAGE_ID >= len(all_cam_params[cam_name]['K']): raise IndexError("Index out of bounds")
                K = all_cam_params[cam_name]['K'][IMAGE_ID]
                R = all_cam_params[cam_name]['R'][IMAGE_ID]
                t = all_cam_params[cam_name]['t'][IMAGE_ID]
                if K is None or R is None or t is None: raise ValueError("Params are None")
                RT = calc_pose_matrix(R, t)
                # Create Camera object, passing rgb_img (which might be None)
                cameras[cam_name] = Camera(name=cam_name, pose=RT, intrinsics=K, rgb=rgb_img, depth=depth_img)
                print(f"    + Successfully created Camera object for {cam_name} (Image loaded: {'Yes' if rgb_img is not None else 'No'})")
            except (IndexError, ValueError, KeyError) as e:
                print(f"    - Error accessing/processing parameters for {cam_name} at index {IMAGE_ID}: {e}. Skipping object creation.")
            except Exception as e:
                print(f"    - Unexpected error creating Camera object for {cam_name}: {e}. Skipping.")


        # --- Step 7: Check for required cameras and get objects ---
        print("\nChecking for required cameras (cam1, cam2, cam3)...")
        cam_1 = cameras.get("cam1")
        cam_2 = cameras.get("cam2")
        cam_3 = cameras.get("cam3")
        photoneo_cam = cameras.get("photoneo")

        required_cams_found = True
        if not cam_1: print("  - Warning: Camera object 'cam1' not created."); required_cams_found = False
        if not cam_2: print("  - Warning: Camera object 'cam2' not created."); required_cams_found = False
        if not cam_3: print("  - Warning: Camera object 'cam3' not created."); required_cams_found = False

        if not required_cams_found:
            print("\nOne or more required cameras (cam1, cam2, cam3) failed object creation. Cannot call get_pose_estimates.")
        else:
            print("  + Required camera objects (cam1, cam2, cam3) created. Proceeding.")
            print(f"  Photoneo camera object created: {'Yes' if photoneo_cam else 'No'}")

            # --- Step 8: Initialize "Estimator" (Verification Mode) ---
            estimator = StandalonePoseEstimator() # No args needed

            # --- Step 9: Call get_pose_estimates for Verification ---
            print("\n--- Calling get_pose_estimates ---")
            # This will now print params and show images internally
            pose_estimates = estimator.get_pose_estimates(
                object_ids=OBJECT_IDS_TO_TEST,
                cam_1=cam_1,
                cam_2=cam_2,
                cam_3=cam_3,
                photoneo=photoneo_cam
            )
            print("\n--- Pose Estimation Results ---")
            if not pose_estimates:
                print("No poses estimated.")
            else:
                for estimate in pose_estimates:
                    print(f"Object ID: {estimate['obj_id']}")
                    print(f"  Score: {estimate['score']:.4f}")
                    pose_mat = estimate['pose']
                    with np.printoptions(precision=3, suppress=True):
                        print(f"  Pose (4x4 Matrix):\n{pose_mat}")
                    try:
                        quat = rot_to_quat(pose_mat[:3, :3])
                        print(f"  Orientation (Quat xyzw): [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                        pos = pose_mat[:3, 3]
                        print(f"  Position (xyz): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    except Exception as e:
                        print(f"  Could not extract quat/pos from pose matrix: {e}")
                    print("-" * 20)


        # except FileNotFoundError as e:
        #     print(f"\nError: A required file or directory was not found.")
        #     print(e)
        # except (ValueError, IndexError, json.JSONDecodeError) as e:
        #     print(f"\nError processing data:")
        #     print(e)
        # except ImportError as e:
        #     print(f"\nImport Error: {e}")
        # except Exception as e:
        #     print(f"\nAn unexpected error occurred: {e}")
        #     import traceback
        #     traceback.print_exc()

        print("\n--- Test Script Finished ---")
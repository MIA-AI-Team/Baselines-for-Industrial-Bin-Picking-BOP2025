import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation # Still needed for calc_pose_matrix if R,t loaded
import sys
from typing import List, Optional, Dict, Any, Tuple
import glob
import json
import re # For extracting camera name from filename
import matplotlib.pyplot as plt
from pose_interface import run_sam6d_pipeline
import argparse
from pose_msgs import Pose, PoseEstimateMsg
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
        output_suffix: str = "",
    ) -> List[PoseEstimateMsg]:
        """
        Estimates poses for given object IDs using data from cam_1, cam_2, cam_3.
        Returns a list of PoseEstimateMsg objects.
        
        Args:
            object_ids: List of object IDs to process
            cam_1, cam_2, cam_3: Required camera objects
            photoneo: Optional photoneo camera
            output_suffix: Suffix to add to output directory names (e.g., image ID)
        """
        pose_estimates_results = []
        # Explicit check for the required cameras passed to this function
        if not all([cam_1, cam_2, cam_3]):
            missing = []
            if not cam_1: missing.append("cam_1")
            if not cam_2: missing.append("cam_2")
            if not cam_3: missing.append("cam_3")
            print(f"Error: get_pose_estimates requires valid Camera objects for {missing}.")
            raise ValueError(f"Missing required Camera objects for pose estimation: {missing}")

        for object_id in object_ids:
            print(f"Running SAM-6D pipeline for object_id: {object_id}")
            # template dir is dataset_dir/templates/obj_000000 where 000000 is the object_id
            template_dir = os.path.join(DATASET_DIR, "templates", f"obj_{object_id:06d}")
            ply_obj_path = os.path.join(DATASET_DIR, "models", f"obj_{object_id:06d}.ply")
            
            # Add image ID suffix to output directory if provided
            if output_suffix:
                output_dir = os.path.join(DATASET_DIR, "results", f"obj_{object_id:06d}_{output_suffix}")
            else:
                output_dir = os.path.join(DATASET_DIR, "results", f"obj_{object_id:06d}")
                
            if not os.path.exists(template_dir):
                print(f"Warning: Template directory does not exist: {template_dir}")
                continue
            if not os.path.exists(ply_obj_path):
                print(f"Warning: Object model file does not exist: {ply_obj_path}")
                continue
            
            # Run SAM-6D pipeline for object detection and pose estimation
            detections, _ = run_sam6d_pipeline(
                camera=cam_1,  # Using only cam_1
                template_dir=template_dir,
                ply_obj_path=ply_obj_path,
                output_dir=output_dir,
                segmentor_model="fastsam",
                stability_score_thresh=0.97,
                det_score_thresh=0.37
            )
            
            # Process the detections to create PoseEstimateMsg objects
            if detections:
                for det in detections:
                    # Check if detection has required pose information
                    if 'R' in det and 't' in det and 'score' in det:
                        # Construct 4x4 transformation matrix
                        R = np.array(det['R'], dtype=np.float64)
                        t = np.array(det['t'], dtype=np.float64) / 1000.0  # Convert to meters if in mm
                        
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = R
                        pose_matrix[:3, 3] = t
                        
                        # Create PoseEstimateMsg
                        pose_msg = PoseEstimateMsg(
                            obj_id=object_id,
                            score=float(det['score']),
                            pose=Pose.from_matrix(pose_matrix)
                        )
                        
                        pose_estimates_results.append(pose_msg)
                        print(f"  Added pose estimate for obj {object_id} with score {det['score']:.3f}")

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

def process_single_image(
    image_id: int, 
    scene_dir: str,
    object_ids: List[int],
    discovered_cam_names: List[str]
) -> Tuple[bool, Dict[str, List[PoseEstimateMsg]]]:
    """
    Process a single image from the scene.
    
    Args:
        image_id: ID of the image to process
        scene_dir: Path to the scene directory
        object_ids: List of object IDs to test
        discovered_cam_names: List of camera names discovered in the scene
        
    Returns:
        success: Whether processing was successful
        results: Dictionary mapping image ID to list of pose estimates
    """
    image_id_str = f"{image_id:06d}"
    print(f"\n\n----- Processing Image ID: {image_id} (Filename: {image_id_str}) -----")
    
    # Load images for this specific image ID
    rgb_images = load_images(scene_dir, discovered_cam_names, image_id_str, "rgb")
    depth_images = load_images(scene_dir, discovered_cam_names, image_id_str, "depth")
    
    # Create Camera objects
    print(f"\nCreating Camera objects for image {image_id}...")
    cameras = {}
    for cam_name in discovered_cam_names:
        print(f"  Processing camera: {cam_name}")
        rgb_img = rgb_images.get(cam_name)
        depth_img = depth_images.get(cam_name)
        
        try:
            # Get camera parameters for this image ID
            all_cam_params = load_camera_params(scene_dir)
            if cam_name not in all_cam_params or not all_cam_params[cam_name]['K']:
                print(f"    - Parameters not loaded. Skipping object creation.")
                continue
                
            if image_id >= len(all_cam_params[cam_name]['K']): 
                raise IndexError("Index out of bounds")
                
            K = all_cam_params[cam_name]['K'][image_id]
            R = all_cam_params[cam_name]['R'][image_id]
            t = all_cam_params[cam_name]['t'][image_id]
            
            if K is None or R is None or t is None: 
                raise ValueError("Params are None")
                
            RT = calc_pose_matrix(R, t)
            cameras[cam_name] = Camera(name=cam_name, pose=RT, intrinsics=K, rgb=rgb_img, depth=depth_img)
            print(f"    + Successfully created Camera object for {cam_name} (Image loaded: {'Yes' if rgb_img is not None else 'No'})")
        except (IndexError, ValueError, KeyError) as e:
            print(f"    - Error accessing parameters for {cam_name} at index {image_id}: {e}")
        except Exception as e:
            print(f"    - Unexpected error creating Camera object for {cam_name}: {e}")
    
    # Check for required cameras
    print(f"\nChecking for required cameras for image {image_id}...")
    cam_1 = cameras.get("cam1")
    cam_2 = cameras.get("cam2")
    cam_3 = cameras.get("cam3")
    photoneo_cam = cameras.get("photoneo")
    
    required_cams_found = all([cam_1, cam_2, cam_3])
    if not required_cams_found:
        print("  - Missing one or more required cameras. Skipping pose estimation.")
        return False, {}
    
    print("  + All required cameras found. Running pose estimation...")
    
    # Initialize estimator and run pose estimation
    estimator = StandalonePoseEstimator()
    pose_estimates = estimator.get_pose_estimates(
        object_ids=object_ids,
        cam_1=cam_1,
        cam_2=cam_2,
        cam_3=cam_3,
        photoneo=photoneo_cam,
        output_suffix=image_id_str
    )
    
    # Store results for this image
    results = {image_id_str: pose_estimates}
    
    return True, results

if __name__ == "__main__":
    # --- Argument Parser ---

    parser = argparse.ArgumentParser(description="Run pose estimation on a dataset.")
    parser.add_argument('--dataset_dir', type=str, default="/content/drive/MyDrive/bpc_opencv_dataset/ipd",
                        help="Path to the dataset directory")
    parser.add_argument('--scene_id', type=str, default="000008",
                        help="Scene ID (e.g., 000008)")
    parser.add_argument('--object_ids', type=int, nargs='+', default=[14],
                        help="List of object IDs to test (e.g., 14)")
    
    # New options for multiple image processing
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument('--image_ids', type=int, nargs='+',
                            help="List of specific image IDs to process (e.g., 0 1 2)")
    image_group.add_argument('--image_range', type=int, nargs=2,
                            help="Range of image IDs to process [start end] (inclusive)")
    image_group.add_argument('--image_id', type=int, default=0,
                            help="Single image ID to process (default: 0)")

    args = parser.parse_args()

    # --- Set Variables from Arguments ---
    DATASET_DIR = args.dataset_dir
    MODEL_DIR = os.path.join(DATASET_DIR, "models")
    SCENE_ID = args.scene_id
    OBJECT_IDS_TO_TEST = args.object_ids

    # Determine which image IDs to process
    if args.image_ids:
        image_ids_to_process = args.image_ids
        print(f"Processing specific image IDs: {image_ids_to_process}")
    elif args.image_range:
        start, end = args.image_range
        image_ids_to_process = list(range(start, end + 1))
        print(f"Processing image ID range: {start} to {end}")
    else:
        image_ids_to_process = [args.image_id]
        print(f"Processing single image ID: {args.image_id}")

    # --- Step 3: Prepare Paths ---
    scene_dir = os.path.join(DATASET_DIR, "test", SCENE_ID)

    print("--- Starting Data Loading Verification via get_pose_estimates ---")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Scene Directory: {scene_dir}")
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

        # --- Process each image ---
        all_results = {}
        
        for image_id in image_ids_to_process:
            success, results = process_single_image(
                image_id=image_id,
                scene_dir=scene_dir,
                object_ids=OBJECT_IDS_TO_TEST,
                discovered_cam_names=discovered_cam_names
            )
            
            if success:
                all_results.update(results)
        
        # --- Output Summary ---
        print("\n=== Overall Results Summary ===")
        if not all_results:
            print("No successful pose estimations.")
        else:
            for image_id, pose_estimates in all_results.items():
                print(f"\nImage ID: {image_id}")
                if not pose_estimates:
                    print("  No poses estimated.")
                else:
                    for i, estimate in enumerate(pose_estimates):
                        print(f"  Detection {i+1}:")
                        print(f"    Object ID: {estimate.obj_id}")
                        print(f"    Score: {estimate.score:.4f}")
                        print(f"    Position: [{estimate.pose.position.x:.3f}, {estimate.pose.position.y:.3f}, {estimate.pose.position.z:.3f}]")
                        print(f"    Orientation (Quat xyzw): [{estimate.pose.orientation.x:.3f}, {estimate.pose.orientation.y:.3f}, "
                              f"{estimate.pose.orientation.z:.3f}, {estimate.pose.orientation.w:.3f}]")
        
        # --- Save Results to JSON ---
        results_dir = os.path.join(DATASET_DIR, "results", f"scene_{SCENE_ID}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save combined results
        combined_results_file = os.path.join(results_dir, "combined_pose_results.json")
        
        # Convert results to serializable format
        serializable_results = {}
        for image_id, pose_estimates in all_results.items():
            serializable_results[image_id] = []
            for est in pose_estimates:
                serializable_results[image_id].append({
                    "obj_id": est.obj_id,
                    "score": est.score,
                    "pose": {
                        "position": {
                            "x": est.pose.position.x,
                            "y": est.pose.position.y,
                            "z": est.pose.position.z
                        },
                        "orientation": {
                            "x": est.pose.orientation.x,
                            "y": est.pose.orientation.y,
                            "z": est.pose.orientation.z,
                            "w": est.pose.orientation.w
                        }
                    }
                })
        
        with open(combined_results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {combined_results_file}")
        print("\n--- Test Script Finished ---")
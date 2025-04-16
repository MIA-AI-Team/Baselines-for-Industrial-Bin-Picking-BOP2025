# %% Imports
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

# --- Modified PoseEstimator Class (No Pose Estimation Logic) ---
class StandalonePoseEstimator:
    def __init__(self):
        # No model loading needed for verification
        print("Initializing StandalonePoseEstimator (Verification Mode - No Models Loaded)")
        pass # No model_dir or model_cache needed

    def get_pose_estimates(
        self,
        object_ids: List[int], # Keep signature, but not used
        cam_1: Camera,
        cam_2: Camera,
        cam_3: Camera,
        photoneo: Optional[Camera] = None,
    ) -> List[Dict[str, Any]]: # Return empty list to match signature
        """
        VERIFICATION FUNCTION: Prints parameters and displays images for input cameras.
        Does NOT perform pose estimation.
        """
        print("\n--- Inside get_pose_estimates (Verification Mode) ---")
        print(f"Object IDs received: {object_ids} (Not used in verification)")

        cams_to_verify = {"cam_1": cam_1, "cam_2": cam_2, "cam_3": cam_3}
        if photoneo:
            cams_to_verify["photoneo"] = photoneo

        valid_cams_count = sum(1 for cam in cams_to_verify.values() if cam is not None)
        if valid_cams_count == 0:
            print("No valid camera objects received. Cannot verify.")
            return []

        plt.figure(figsize=(15, 5 * ((valid_cams_count + 1) // 2))) # Adjust figure size
        plot_index = 1

        for name, cam in cams_to_verify.items():
            print(f"\n--- Verifying Data for: {name} ---")
            if cam is None:
                print("  Camera object is None. Skipping.")
                continue

            # Print parameters stored in the Camera object
            print(f"  Camera Name: {cam.name}") # Should match the key 'name'
            print("  Intrinsics (K):")
            with np.printoptions(precision=3, suppress=True):
                print(f"  {cam.intrinsics}")
            print("\n  Pose Matrix (World-to-Camera):")
            with np.printoptions(precision=3, suppress=True):
                print(f"  {cam.pose}")

            # Display the image
            if cam.rgb is not None:
                print(f"\n  Displaying RGB Image (shape: {cam.rgb.shape})...")
                try:
                    # Convert BGR (from cv2) to RGB (for matplotlib)
                    rgb_img_rgb = cv2.cvtColor(cam.rgb, cv2.COLOR_BGR2RGB)
                    plt.subplot((valid_cams_count + 1) // 2, 2, plot_index)
                    plt.imshow(rgb_img_rgb)
                    plt.title(f"Input: {name} ({cam.name})")
                    plt.axis('off')
                    plot_index += 1
                except Exception as e:
                    print(f"  Error displaying image for {name}: {e}")
            else:
                print("  RGB Image data is None. Cannot display.")

        if plot_index > 1:
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo images were available to display from the input cameras.")

        print("\n--- Exiting get_pose_estimates (Verification Mode) ---")
        return [] # Return empty list as no poses were estimated

# --- Data Loading Functions (Unchanged) ---

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

def load_images(scene_dir: str, cam_names: List[str], image_id_str: str) -> Dict[str, Optional[np.ndarray]]:
    """Loads RGB images for the given camera names and image ID. Returns None if loading fails."""
    images = {}
    print(f"\nLoading images for image ID: {image_id_str}")
    for cam_name in cam_names:
        folder_name = f"rgb_{cam_name}"
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

# --- Main Execution for Data Loading and Verification via get_pose_estimates ---

# --- Step 1: Mount Google Drive (Run in a separate cell first) ---
# from google.colab import drive
# drive.mount('/content/drive')

# --- Step 2: Configuration ---
# !!! ADJUST PATHS FOR YOUR GOOGLE DRIVE SETUP !!!
DATASET_DIR = "/content/drive/MyDrive/path/to/your/ipd" # CHANGE THIS
SCENE_ID = "000008"
IMAGE_ID = 0
OBJECT_IDS_TO_TEST = [1] # Still need to pass something, value doesn't matter now

# --- Step 3: Prepare Paths ---
scene_dir = os.path.join(DATASET_DIR, "test", SCENE_ID)
image_id_str = f"{IMAGE_ID:06d}"

print("--- Starting Data Loading Verification via get_pose_estimates ---")
print(f"Dataset Directory: {DATASET_DIR}")
print(f"Scene Directory: {scene_dir}")
print(f"Image Index: {IMAGE_ID} (Filename ID: {image_id_str})")

if not os.path.isdir(scene_dir):
    print(f"\nError: Scene directory not found: {scene_dir}")
else:
    try:
        # --- Step 4: Discover cameras and load parameters ---
        print("\nDiscovering cameras and loading parameters...")
        all_cam_params = load_camera_params(scene_dir)
        discovered_cam_names = sorted(list(all_cam_params.keys()))
        if not discovered_cam_names: raise FileNotFoundError(f"No camera param files found in {scene_dir}.")
        print(f"Discovered cameras: {discovered_cam_names}")

        # --- Step 5: Load Images for the SPECIFIC image_id ---
        images = load_images(scene_dir, discovered_cam_names, image_id_str)

        # --- Step 6: Create Camera Objects ---
        print("\nCreating Camera objects...")
        cameras = {}
        for cam_name in discovered_cam_names:
            print(f"  Processing camera: {cam_name}")
            rgb_img = images.get(cam_name) # Get image array or None
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
                cameras[cam_name] = Camera(name=cam_name, pose=RT, intrinsics=K, rgb=rgb_img)
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
            # This will now print params and show images internally
            verification_results = estimator.get_pose_estimates(
                object_ids=OBJECT_IDS_TO_TEST,
                cam_1=cam_1,
                cam_2=cam_2,
                cam_3=cam_3,
                photoneo=photoneo_cam
            )
            # verification_results will be an empty list []
            print(f"\nCall to get_pose_estimates completed. Verification results (should be empty): {verification_results}")


    except FileNotFoundError as e: print(f"\nError: {e}")
    except (ValueError, IndexError, json.JSONDecodeError) as e: print(f"\nError processing data: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

print("\n--- Verification Script Finished ---")
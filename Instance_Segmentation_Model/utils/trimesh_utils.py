import numpy as np
import trimesh
import torch


def load_mesh(path, ORIGIN_GEOMETRY="BOUNDS"):
    mesh = as_mesh(trimesh.load(path))
    if ORIGIN_GEOMETRY == "BOUNDS":
        AABB = mesh.bounds
        center = np.mean(AABB, axis=0)
        mesh.vertices -= center
    return mesh


def get_bbox_from_mesh(mesh):
    AABB = mesh.bounds
    OBB = AABB_to_OBB(AABB)
    return OBB


def get_obj_diameter(mesh_path):
    mesh = load_mesh(mesh_path)
    extents = mesh.extents * 2
    return np.linalg.norm(extents)


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        result = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()
            ]
        )
    else:
        result = scene_or_mesh
    return result


def AABB_to_OBB(AABB):
    """
    AABB bbox to oriented bounding box
    """
    minx, miny, minz, maxx, maxy, maxz = np.arange(6)
    corner_index = np.array(
        [
            minx,
            miny,
            minz,
            maxx,
            miny,
            minz,
            maxx,
            maxy,
            minz,
            minx,
            maxy,
            minz,
            minx,
            miny,
            maxz,
            maxx,
            miny,
            maxz,
            maxx,
            maxy,
            maxz,
            minx,
            maxy,
            maxz,
        ]
    ).reshape((-1, 3))

    corners = AABB.reshape(-1)[corner_index]
    return corners

def depth_image_to_pointcloud_translate_torch(depth, scale, K):
    batch_size = depth.shape[0]
    height = depth.shape[1]
    width = depth.shape[2]
    
    # Initialize results
    sum_X = torch.zeros(batch_size, device=depth.device)
    sum_Y = torch.zeros(batch_size, device=depth.device)
    sum_Z = torch.zeros(batch_size, device=depth.device)
    valid_counts = torch.zeros(batch_size, device=depth.device)
    
    # Process in batch chunks to reduce memory usage
    for b_idx in range(batch_size):
        # Get single depth map
        single_depth = depth[b_idx:b_idx+1]  # Keep dimension
        
        # Process in row chunks
        chunk_size = 200  # Adjust based on memory constraints
        for start_row in range(0, height, chunk_size):
            end_row = min(start_row + chunk_size, height)
            
            # Create coordinate grid for this chunk
            v_chunk = torch.arange(start_row, end_row).to(depth.device)
            u_chunk = torch.arange(0, width).to(depth.device)
            v_grid, u_grid = torch.meshgrid(v_chunk, u_chunk, indexing="ij")  # Shape: [chunk_height, width]

            depth_chunk = single_depth[:, start_row:end_row, :]  # [1, chunk_height, width]

            Z_chunk = depth_chunk * scale / 1000
            X_chunk = (u_grid - K[0, 2]) * Z_chunk / K[0, 0]
            Y_chunk = (v_grid - K[1, 2]) * Z_chunk / K[1, 1]
            
            valid_chunk = Z_chunk > 0
            
            # Apply validity mask
            X_valid = X_chunk * valid_chunk
            Y_valid = Y_chunk * valid_chunk
            Z_valid = Z_chunk * valid_chunk
            
            # Accumulate sums and counts for this batch item
            sum_X[b_idx] += torch.sum(X_valid)
            sum_Y[b_idx] += torch.sum(Y_valid)
            sum_Z[b_idx] += torch.sum(Z_valid)
            valid_counts[b_idx] += torch.count_nonzero(valid_chunk)
            
            # Clear memory
            del u_grid, v_grid, depth_chunk, Z_chunk, X_chunk, Y_chunk
            del X_valid, Y_valid, Z_valid, valid_chunk
            torch.cuda.empty_cache()
    
    # Compute averages
    valid_counts = valid_counts + 1e-8  # Avoid division by zero
    avg_X = sum_X / valid_counts
    avg_Y = sum_Y / valid_counts
    avg_Z = sum_Z / valid_counts
    
    # Create final result
    translate = torch.stack((avg_X, avg_Y, avg_Z), dim=1)
    
    return translate

if __name__ == "__main__":
    mesh_path = (
        "/media/nguyen/Data/dataset/ShapeNet/ShapeNetCore.v2/"
        "03001627/1016f4debe988507589aae130c1f06fb/models/model_normalized.obj"
    )
    mesh = load_mesh(mesh_path)
    bbox = get_bbox_from_mesh(mesh)
    # create a visualization scene with rays, hits, and mesh
    scene = trimesh.Scene([mesh, trimesh.points.PointCloud(bbox)])
    # display the scene
    scene.show()

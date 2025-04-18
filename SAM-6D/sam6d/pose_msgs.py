import numpy as np
from scipy.spatial.transform import Rotation

class Point:
    """Mimics ROS geometry_msgs/Point structure."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __str__(self):
        return f"Point(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"
    
    @classmethod
    def from_array(cls, array):
        """Create a Point from a numpy array."""
        if len(array) != 3:
            raise ValueError(f"Point array must have 3 elements, got {len(array)}")
        return cls(x=array[0], y=array[1], z=array[2])

class Quaternion:
    """Mimics ROS geometry_msgs/Quaternion structure."""
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
    
    def __str__(self):
        return f"Quaternion(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, w={self.w:.3f})"
    
    @classmethod
    def from_rotation_matrix(cls, rot_matrix):
        """Create a Quaternion from a 3x3 rotation matrix."""
        r = Rotation.from_matrix(rot_matrix)
        quat = r.as_quat()  # Returns [x, y, z, w]
        return cls(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

class Pose:
    """Mimics ROS geometry_msgs/Pose structure."""
    def __init__(self, position=None, orientation=None):
        self.position = position if position else Point()
        self.orientation = orientation if orientation else Quaternion()
    
    def __str__(self):
        return f"Pose(\n  {self.position}\n  {self.orientation}\n)"
    
    @classmethod
    def from_matrix(cls, matrix):
        """Create a Pose from a 4x4 transformation matrix."""
        if matrix.shape != (4, 4):
            raise ValueError(f"Pose matrix must be 4x4, got {matrix.shape}")
        
        position = Point.from_array(matrix[:3, 3])
        orientation = Quaternion.from_rotation_matrix(matrix[:3, :3])
        
        return cls(position=position, orientation=orientation)

class PoseEstimateMsg:
    """Mimics a ROS pose estimation message."""
    def __init__(self, obj_id=0, score=0.0, pose=None):
        self.obj_id = int(obj_id)
        self.score = float(score)
        self.pose = pose if pose else Pose()
    
    def __str__(self):
        return f"PoseEstimateMsg(\n  obj_id={self.obj_id}\n  score={self.score:.3f}\n  {self.pose}\n)"

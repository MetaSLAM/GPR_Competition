# Loading Data
We provide data loader for the datasets. With these loaders, you can easily get access to the data (point clouds, images, etc.) and the poses of each frame/submap.

Data type of each frame:
-    LiDAR point clouds: open3d.geometry.PointCloud
-    Image: PIL.Image object
-    pose: T = numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
-    rotation: R $\in$ SO(3), numpy.ndarray, of size (3, 3)
-    translation: t = numpy.ndarray([x, y, z]), of size (3, 1)

We provide **sample data** for quick testing. The sample data for the Pittsburgh City-scale Dataset can be found [here](https://sandbox.zenodo.org/record/1033096).

## class `PittsLoader`
```python
class PittsLoader(BaseLoader):
    def __len__(self) -> int:
        """Return the number of frames in this dataset"""

    def __str__(self) -> str:
        return f'PittsLoader at "{self.dir_path}" with {self.len} submaps.'

    def __repr__(self) -> str:
        return f'PittsLoader at "{self.dir_path}" with {self.len} submaps.'

    def __getitem__(self, frame_id: int):
        """Return the query data (Image, LiDAR, etc)
        Args:
            frame_id: the index of current frame
        Returns:
            data: Dict['img':Image, 'pcd':LiDAR, ...]
        """

    def get_point_cloud(self, frame_id: int) -> np.ndarray:
        """Get the point cloud at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pcd: N*3 point clouds
        """
    
    def get_pose(self, frame_id: int) -> np.ndarray:
        """Get the pose (4*4 transformation matrix) at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pose: numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        Raise:
            ValueError: If this dataset doesn't have poses
        """

    def get_rotation(self, frame_id: int, type: str = 'matrix') -> np.ndarray:
        """Get the rotation part at of the pose at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
            type: can be one of {'matrix', 'rpy', 'quat'}. 'matrix'-> 3*3 rotation matrix,
                'rpy'-> roll, pitch, yaw angles, 'quat'-> quaternion
        Returns:
            rotation: if type == 'matrix', then it is 3*3 rotation matrix. If type == 'rpy',
                then it is (roll, pitch, yaw) of size (3,). If type == 'quat', then it is
                quaternion (qx, qy, qz, qw) of size (4,).
        Raises:
            ValueError: if type is not one of {'matrix', 'rpy', 'quat'}.
        """

    def get_translation(self, frame_id: int) -> np.ndarray:
        """Get the 3*1 translation vector of the pose at the `frame_id` frame
        Args:
            frame_id: the index of current frame
        Returns:
            translation: (3,) np.ndarray, the translation vector.
        """
```

# Examples
## Get point clouds and its pose
```python
from gpr.dataloader import PittsLoader

dataset_path = 'PATH_TO_THE_DATASET'
pitts_loader = PittsLoader(dataset_path)

submap_id = 50
pcd_ndarray = pitts_loader[submap_id]['pcd'] 
# or pcd_ndarray = pitts_loader.get_point_cloud(submap_id)
pose = pitts_loader.get_pose(submap_id)
```

## Iterate the dataloader
```python
from gpr.dataloader import PittsLoader

dataset_path = 'PATH_TO_THE_DATASET'
pitts_loader = PittsLoader(dataset_path)

for frame_id in range(len(pitts_loader)):
    pcd = pitts_loader.get_point_cloud(frame_id)
    poses = pitts_loader.get_pose(frame_id)
```

## Get the trajectory xyz
```python
import numpy as np
from gpr.dataloader import PittsLoader

dataset_path = 'PATH_TO_THE_DATASET'
pitts_loader = PittsLoader(dataset_path)

trajectory = [
    pitts_loader.get_translation(frame_id) 
    for frame_id in range(len(pitts_loader))
]
trajectory = np.vstack(trajectory)
```
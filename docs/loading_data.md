# Loading Data
We provide data loader for the datasets. These datasets are based on the kapture format. However, the original kapture class only provides file name to recorded data, which is not so convenient. This loader gives interfaces to directly get the data.

Data type of each frame:
-    LiDAR point clouds: open3d.geometry.PointCloud
-    Image: PIL.Image object
-    pose: T = numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
-    rotation: R $\in$ SO(3), numpy.ndarray, of size (3, 3)
-    translation: t = numpy.ndarray([x, y, z]), of size (3, 1)

## class `DataLoader`
```python
class DataLoader:
    def __len__(self) -> None:
        """Return the number of frames in this dataset"""
    
    def get_point_cloud(self, frame_id: int) -> o3d.geometry.PointCloud:
        """Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        """
    
    def get_image(self, frame_id: int) -> PIL.Image.Image:
        """Get the image at the `frame_id` frame.
        Raise ValueError if there is no image in the dataset.
        """
    
    def get_rotation(self, frame_id: int) -> np.ndarray:
        """Get the 3*3 rotation matrix at the `frame_id` frame."""
    
    def get_translation(self, frame_id: int) -> np.ndarray:
        """Get the 3*1 translation vector at the `frame_id` frame"""
    
    def get_pose(self, frame_id: int) -> np.ndarray:
        """Get the pose (transformation matrix) at the `frame_id` frame.
        numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        """
```

## module `metaslam`, public function
```python
def load_dataset(dataset_path: Union[str, Path]) -> List[DataLoader]:
    """Load the dataset with all its trajectories.
    Each trajectory is represented as a `DataLoader` object, and they are
    sorted with the trajectory folder name in the list.
    """
```

# Examples
## Get point clouds and convert to numpy.ndarray
```python
import metaslam
import numpy as np

dataset_path = 'PATH_TO_THE_DATASET'
dataset = metaslam.load_dataset(dataset_path) # List[traj1, traj2, ...]

traj0 = dataset[0]    # select the first trajectory, DataLoader object
frame_id = 88         # get the data of 88th frame
pcd = traj0.get_point_cloud(frame_id) # open3d.geometry.PointCloud
pcd_np = np.asarray(pcd.points)       # np.ndarray
```

## Iterate a certain trajectory
```python
import metaslam

dataset_path = 'PATH_TO_THE_DATASET'
dataset = metaslam.load_dataset(dataset_path) # List[traj1, traj2, ...]
traj0 = dataset[0]  # select the first trajectory, DataLoader object

for frame_id in range(len(traj0)):
    pcd = traj0.get_point_cloud(frame_id)
    poses = traj0.get_pose(frame_id)
```

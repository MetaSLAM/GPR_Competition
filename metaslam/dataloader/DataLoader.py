"""Data loader for the datasets.
These datasets are based on the kapture format. However, the original
kapture class only provides file name to recorded data, which is not so
convenient. This loader provides interfaces to directly get the data.

Data type of each frame:
    LiDAR point clouds: open3d.geometry.PointCloud
    Image: PIL.Image object
    pose: T = numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
    rotation: R \in SO(3), numpy.ndarray, of size (3, 3)
    translation: t = numpy.ndarray([x, y, z]), of size (3, 1)
"""

from pathlib import Path
import PIL.Image
import numpy as np
import open3d as o3d
from typing import List, Union

import kapture.io.csv as csv
from kapture.core.PoseTransform import _as_rotation_matrix_njit


class DataLoader:
    def __init__(self, kapture_dir_path: Union[str, Path]):
        self._kapture_data = csv.kapture_from_dir(kapture_dir_path)
        self._records_camera = self._kapture_data.records_camera
        self._records_lidar = self._kapture_data.records_lidar
        self._trajectories = self._kapture_data.trajectories

        self._kapture_dir_path = Path(kapture_dir_path)
        self._records_data_path = self._kapture_dir_path / Path("sensors/records_data")

    def __len__(self):
        """Return the number of frames in this dataset"""
        if self._records_camera is not None:
            return len(self._records_camera)
        else:
            return len(self._records_lidar)

    def get_point_cloud(self, frame_id: int) -> o3d.geometry.PointCloud:
        """Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        """
        if self._records_lidar is None:
            raise ValueError("This dataset does NOT have point cloud data")

        pcd_path = self._records_data_path / Path(
            self._records_lidar[(frame_id, "lidar0")]
        )
        return o3d.io.read_point_cloud(str(pcd_path))

    def get_image(self, frame_id: int) -> PIL.Image.Image:
        """Get the image at the `frame_id` frame.
        Raise ValueError if there is no image in the dataset.
        """
        if self._records_camera is None:
            raise ValueError("This dataset does NOT have image data")

        image_path = self._records_data_path / Path(
            self._records_camera[(frame_id, "camera0")]
        )
        return PIL.Image.open(image_path)

    def get_rotation(self, frame_id: int) -> np.ndarray:
        """Get the 3*3 rotation matrix at the `frame_id` frame."""
        r_kapture = self._trajectories[(frame_id, "rig")].r
        r_np = np.array([r_kapture.w, r_kapture.x, r_kapture.y, r_kapture.z])
        r_matrix = np.empty((3, 3), dtype=float)
        _as_rotation_matrix_njit(r_np, r_matrix)

        return r_matrix

    def get_translation(self, frame_id: int) -> np.ndarray:
        """Get the 3*1 translation vector at the `frame_id` frame"""
        return self._trajectories[(frame_id, "rig")].t

    def get_pose(self, frame_id: int) -> np.ndarray:
        """Get the pose at the `frame_id` frame.
        Transformation matrix, numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        """
        T_4x4 = np.identity(4, dtype=float)
        T_4x4[:3, :3] = self.get_rotation(frame_id)
        T_4x4[:3, 3:] = self.get_translation(frame_id)
        return T_4x4

    def __repr__(self) -> str:
        traj_name = self._kapture_dir_path.stem
        frame_num = self.__len__()
        return f"<DataLoader of '{traj_name}' with {frame_num} frames>"


def load_dataset(dataset_path: Union[str, Path]) -> List[DataLoader]:
    """Load the dataset with all its trajectories.
    Each trajectory is represented as a `DataLoader` object, and they are
    sorted with the trajectory folder name in the list.
    """
    dataset_path = Path(dataset_path)
    trajs_path = list(dataset_path.iterdir())
    trajs_path.sort()

    dataset = [DataLoader(traj_path) for traj_path in trajs_path]
    return dataset

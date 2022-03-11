"""
Created Date: Thursday, March 10th 2022, 9:26:53 pm
Author: Haowen Lai

Copyright (c) 2022 Your Company
"""

import open3d as o3d
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R

from .BaseLoader import BaseLoader


class PittsLoader(BaseLoader):
    def __init__(self, dir_path: str):
        """Data loader for the Pittsburgh Dataset."""
        super().__init__(dir_path)

        self.len = len(glob(self.dir_path + '/*.pcd'))

    def __len__(self) -> int:
        """Return the number of frames in this dataset"""
        return self.len

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
        pcd = self.get_point_cloud(frame_id)
        return {'pcd': pcd}

    def get_pose(self, frame_id: int) -> np.ndarray:
        """Get the pose (4*4 transformation matrix) at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pose: numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        """
        pose6d = np.load(self.dir_path + f'/{frame_id:06d}_pose6d.npy')[:6]
        rot_matrix = R.from_euler('xyz', pose6d[3:]).as_matrix()
        trans_vector = pose6d[:3].reshape((3, 1))

        trans_matrix = np.identity(4)
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[:3, 3:] = trans_vector

        return trans_matrix

    def get_image(self, frame_id: int):
        """Get the image at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            image: PIL.Image image
        """
        raise ValueError('This dataset does NOT have images.')

    def get_point_cloud(self, frame_id: int) -> np.ndarray:
        """Get the point cloud at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pcd: N*3 point clouds
        """
        pcd = o3d.io.read_point_cloud(self.dir_path + f'/{frame_id+1:06d}.pcd')
        pcd = np.asarray(pcd.points)
        return pcd

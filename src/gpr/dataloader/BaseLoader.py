'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom, Haowen

Copyright (c) 2022 Your Company
'''

from typing import Dict
import numpy as np
from abc import abstractmethod
from scipy.spatial.transform import Rotation as R


class BaseLoader(object):
    """This class serves as the interface for all data loaders of different datasets"""

    def __init__(self, dir_path: str):
        """BaseLoader initlization"""
        self.dir_path = dir_path

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of frames in this dataset"""
        pass

    @abstractmethod
    def __getitem__(self, frame_id: int) -> Dict:
        """Return the query data (Image, LiDAR, etc)
        Args:
            frame_id: the index of current frame
        Returns:
            data: Dict['img':Image, 'pcd':LiDAR, ...]
        """
        pass

    @abstractmethod
    def get_pose(self, frame_id: int) -> np.ndarray:
        """Get the pose (4*4 transformation matrix) at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pose: numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        Raise:
            ValueError: If this dataset doesn't have poses
        """
        pass

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
        transform = self.get_pose(frame_id)

        if type == 'matrix':
            return transform[:3, :3]
        elif type == 'rpy':
            return R.from_matrix(transform[:3, :3]).as_euler('xyz')
        elif type == 'quat':
            return R.from_matrix(transform[:3, :3]).as_quat()
        else:
            raise ValueError(f'{type} is not a valid type.')

    def get_translation(self, frame_id: int) -> np.ndarray:
        """Get the 3*1 translation vector of the pose at the `frame_id` frame
        Args:
            frame_id: the index of current frame
        Returns:
            translation: (3,) np.ndarray, the translation vector.
        """
        transform = self.get_pose(frame_id)
        return transform[:3, 3]

    @abstractmethod
    def get_point_cloud(self, frame_id: int) -> np.ndarray:
        """Get the point cloud at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pcd: N*3 point clouds
        """
        pass

    @abstractmethod
    def get_image(self, frame_id: int):
        """Get the image at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            image: PIL.Image image
        """
        pass

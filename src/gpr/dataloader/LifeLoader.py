'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
from typing import Tuple
import open3d as o3d
from PIL import Image
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R

from .BaseLoader import BaseLoader
from ..tools import lidar_trans


class LifeLoader(BaseLoader):
    def __init__(
        self,
        dir_path: str,
        image_size: Tuple[int, int] = (512, 512),
        top_size: Tuple[int, int] = (512, 512),
        sph_size: Tuple[int, int] = (512, 512),
        fov_range: Tuple[int, int] = (-90, 90),
        resolution: float = 0.5,
        max_radius: float = 50,
    ):
        """Data loader for Life Dataset.
        Args:
            image_size [int, int]: set image resolution
            top_size [int, int]: set top down view resolution
            sph_size [int, int]: set spherical view resolution
            resolution [float]: resolution for point cloud voxels
            max_radius [int]: maximum distance for sub map
        """
        super().__init__(dir_path)
        self.resolution = resolution
        self.max_radius = max_radius
        self.queries = sorted(glob("{}/*.png".format(self.dir_path)))

        # * for raw RGB image
        # self.image_trans = image_trans(image_size=image_size, channel=3)

        # * for lidar projections
        self.lidar_trans = lidar_trans(
            top_size, sph_size, max_dis=self.max_radius, fov_range=fov_range
        )

        # * Obtain raw point cloud map
        map_pcd = o3d.io.read_point_cloud(self.dir_path + "/dense.pcd")
        self.map_pcd = map_pcd.voxel_down_sample(self.resolution)
        self.tree = o3d.geometry.KDTreeFlann(self.map_pcd)

    def __len__(self) -> int:
        """Return the number of frames in this dataset"""
        return len(self.queries)

    def __getitem__(self, frame_id: int):
        """Return the query data (Image, LiDAR, etc)
        Args:
            frame_id: the index of current frame
        Returns:
            data: Dict['img':Image, 'pcd':LiDAR, ...]
        """
        img = self.get_image(frame_id)
        pcd = self.get_point_cloud(frame_id)
        sph = self.lidar_trans.sph_projection(pcd)  # get spherical projection
        top = self.lidar_trans.top_projection(pcd)  # get top_down projection

        return {'img': img, 'pcd': pcd, 'sph': sph, 'top': top}

    def get_pose(self, frame_id: int) -> np.ndarray:
        """Get the pose (4*4 transformation matrix) at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pose: numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        """
        trans_matrix = np.loadtxt(
            '{}.odom'.format(self.queries[frame_id].split('.')[0])
        )
        return trans_matrix

    def get_image(self, frame_id: int):
        """Get the image at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            image: PIL.Image image
        """
        return Image.open(self.queries[frame_id])

    def get_point_cloud(self, frame_id: int) -> np.ndarray:
        """Get the point cloud at the `frame_id` frame.
        Args:
            frame_id: the index of current frame
        Returns:
            pcd: N*3 point clouds
        """
        trans_matrix = self.get_pose(frame_id, rot_type='matrix')
        translation = self.get_translation(trans_matrix)
        rotation = self.get_rotation(trans_matrix)

        [_, idx, _] = self.tree.search_radius_vector_3d(translation, self.max_radius)

        # * get raw point cloud
        pcd_data = np.asarray(self.map_pcd.points)[idx, :] - translation - 1.5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data)
        rot = R.from_matrix(rotation)
        trans_matrix = np.eye(4)
        trans_matrix[0:3, 0:3] = rot.inv().as_matrix()
        pcd.transform(trans_matrix)
        pcd = np.array(pcd.points)

        return pcd

'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
import open3d as o3d
import numpy as np
from glob import glob
from .BaseLoader import BaseLoader
from ..tools import lidar_trans, image_trans


class LifeLoader(BaseLoader):
    def __init__(self, dir_path, image_size=[512, 512], top_size=[512, 512], sph_size=[512, 512], resolution=0.5, max_radius=50):
        ''' image_size [int, int]: set image resolution
            top_size [int, int]: set top down view resolution
            sph_size [int, int]: set spherical view resolution
            resolution [float]: resolution for point cloud voxels
            max_radius [int]: maxisimum distance for sub map
        '''
        super().__init__(dir_path)
        self.resolution = resolution
        self.max_radius = max_radius

        # * for raw RGB image
        self.image_trans = image_trans(image_size=image_size, channel=3)

        # * for lidar projections
        self.lidar_trans = lidar_trans(top_size, sph_size, max_dis=self.max_radius)

        # * Obtain raw point cloud map
        map_pcd = o3d.io.read_point_cloud(self.dir_path+"/dense.pcd")
        self.map_pcd = map_pcd.voxel_down_sample(self.resolution)
        self.tree = o3d.geometry.KDTreeFlann(self.map_pcd)

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        img = self.get_image(idx)
        pcd, sph, top = self.get_point_cloud(idx)
        return {'img': img, 'pcd': pcd, 'sph': sph, 'top': top}

    def get_query(self):
        ''' Return the query information based on different tasks'''
        return sorted(glob("{}/*.png".format(self.dir_path)))

    def get_image(self, frame_id: int):
        '''Get the image at the `frame_id` frame.
        Raise ValueError if there is no image in the dataset.
        return -> Image.Image
        '''
        return self.image_trans.get_data(self.queries[frame_id])

    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''

        _, translation, _ = self.get_pose(frame_id)

        [_, idx, _] = self.tree.search_radius_vector_3d(translation, self.max_radius)

        # * get raw point cloud
        pcd = np.asarray(self.map_pcd.points)[idx, :] - translation

        # * get spherical projection
        sph = self.lidar_trans.sph_projection(pcd)

        # * get top_down projection
        top = self.lidar_trans.top_projection(pcd)

        return pcd, sph, top

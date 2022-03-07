'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
import open3d as o3d
import numpy as np
from .BaseLoader import BaseLoader
from ..tools import lidar_trans


class UgvLoader(BaseLoader):
    def __init__(self, dir_path, top_size=[512, 512], sph_size=[512, 512], resolution=0.5):
        ''' image_size [int, int]: set image resolution
            resolution [float]: resolution for point cloud voxels
        '''
        super().__init__(dir_path)
        self.resolution = resolution

        # * for lidar projections
        self.lidar_trans = lidar_trans(top_size, sph_size)

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        return self.get_point_cloud(idx)

    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''
        pcd_data = o3d.io.read_point_cloud(self.queries[frame_id])
        ds_pcd = pcd_data.voxel_down_sample(self.resolution)

        #* get raw point cloud
        pcd = np.asarray(ds_pcd.points)

        #* get spherical projection
        sph = self.lidar_trans.sph_projection(pcd)
    
        #* get top_down projection
        top = self.lidar_trans.top_projection(pcd)

        return pcd, sph, top

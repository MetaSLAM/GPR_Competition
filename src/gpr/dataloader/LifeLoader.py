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
    def __init__(self, dir_path, top_size=[512, 512], sph_size=[512, 512], resolution=0.5, radius=50):
        ''' top_size [int, int]: set image resolution
            sph_size [int, int]: set image resolution
            resolution [float]: resolution for point cloud voxels
            radius [int]: maxisimum distance for sub map
        '''
        super().__init__(dir_path)
        self.resolution = resolution
        self.radius = radius

        # * for lidar projections
        self.lidar_trans = lidar_trans(top_size, sph_size, max_dis=radius)

        # * Obtain raw point cloud map
        map_pcd = o3d.io.read_point_cloud(dir_path+"/dense.pcd")
        self.map_pcd = map_pcd.voxel_down_sample(self.resolution)
        self.tree = o3d.geometry.KDTreeFlann(self.map_pcd)

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        return self.get_point_cloud(idx)

    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''
        
        frame_ori = np.load(self.queries[frame_id])

        [k, idx, _] = self.tree.search_radius_vector_3d(frame_ori, self.radius)
        
        #* get raw point cloud
        pcd = np.asarray(self.map_pcd.points)[idx, :] - frame_ori

        #* get spherical projection
        sph = self.lidar_trans.sph_projection(pcd)
    
        #* get top_down projection
        top = self.lidar_trans.top_projection(pcd)

        return pcd, sph, top

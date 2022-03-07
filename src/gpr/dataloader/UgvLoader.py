'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
import open3d as o3d
from .BaseLoader import BaseLoader
from ..tools import lidar_trans


class UgvLoader(BaseLoader):
    def __init__(self, dir_path, image_size=[512, 512], random_rotation=False):
        ''' image_size [int, int]: set image resolution
            random_rotation: True/False, set viewpoint rotation 
        '''
        super().__init__(dir_path)

        # * for lidar projections
        self.lidar_trans = lidar_trans(image_size=image_size, channel=1)

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        return self.get_point_cloud(idx)

    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''
        return self.lidar_trans(self.queries[frame_id])

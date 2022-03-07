'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
import open3d as o3d
from PIL import Image
from .BaseLoader import BaseLoader
from ..tools.geometry import image_trans


class UgvLoader(BaseLoader):
    def __init__(self, dir_path):
        super().__init__(dir_path)

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        return self.get_point_clud(idx)

    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''
        pass
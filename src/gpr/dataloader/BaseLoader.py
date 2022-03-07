'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

import os
import random
import numpy as np
from abc import abstractmethod
from ..tools.geometry import image_trans


class BaseLoader(object):
    def __init__(self, dir_path):
        ''' BaseLoader initlization
        '''
        self.dataset_dir = dir_path

        #* for raw RGB image
        self.image_trans = image_trans(image_size=[64,64], channel=3)

        #* for raw RGB image
        self.lidar_trans = image_trans(image_size=[64,64], channel=3)

        #* obtain queries
        self.queries = self.get_query(self.dataset_dir)

    def __len__(self):
        ''' Return the number of frames in this dataset
        '''
        return len(self.queries)

    def __getitem__(self, idx):
        '''Return the query data (Image, LiDAR, etc)'''
        pass

    def get_query(self, dir_path):
        '''Return data with the trajectory'''
        pass

    def get_rotation(self, frame_id: int):
        '''Get the 3*3 rotation matrix at the `frame_id` frame.
        return -> np.ndarray
        '''
        pass

    def get_translation(self, frame_id: int):
        '''Get the 3*1 translation vector at the `frame_id` frame
        return -> np.ndarray
        '''
        pass

    def get_pose(self, frame_id: int):
        '''Get the pose (transformation matrix) at the `frame_id` frame.
        numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        return -> np.ndarray
        '''
        pass

    @abstractmethod
    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''
        pass

    @abstractmethod
    def get_image(self, frame_id: int):
        '''Get the image at the `frame_id` frame.
        Raise ValueError if there is no image in the dataset.
        return -> Image.Image
        '''
        pass

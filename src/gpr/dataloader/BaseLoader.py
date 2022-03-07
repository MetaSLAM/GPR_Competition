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
from scipy.spatial.transform import Rotation as R


class BaseLoader(object):
    def __init__(self, dir_path):
        ''' BaseLoader initlization
        '''
        self.dir_path = dir_path

        # * obtain queries
        self.queries = self.get_query()

    def __len__(self):
        ''' Return the number of frames in this dataset
        '''
        return len(self.queries)

    def __getitem__(self, idx):
        '''Return the query data (Image, LiDAR, etc)'''
        pass

    def get_pose(self, frame_id: int):
        '''Get the pose (transformation matrix) at the `frame_id` frame.
        numpy.ndarray([[R, t], [0, 1]]), of size (4, 4)
        return -> np.ndarray
        '''
        trans_matrix = np.loadtxt('{}.odom'.format(
            self.queries[frame_id].split('.')[0]))
        translation = self.get_translation(trans_matrix)
        rotation = self.get_rotation(trans_matrix)
        return trans_matrix, translation, rotation

    def get_rotation(self, transform, type='matrix'):
        '''Get the rotation at the `frame_id` frame.
        transform: 4x4 np.ndarray
        type: 'matrix'->3*3 rotation matrix, 'rpy'-> roll, pitch, yaw angles, 'quat'-> quaternion
        '''
        if type == 'matrix':
            return transform[:3, :3]
        else:
            rot = transform[:3, :3]
            r = R.from_matrix(rot)
            if type == 'quat':
                return r.as_quat()
            elif type == 'rpy':
                return r.as_rotvec()

    def get_translation(self, transform):
        '''Get the 3*1 translation vector at the `frame_id` frame
        return -> np.ndarray
        '''
        return transform[:, 3][:3]

    @abstractmethod
    def get_query(self):
        '''Return data with the trajectory'''
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

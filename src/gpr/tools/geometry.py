'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/tools/geometry.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/tools
Created Date: Sunday, March 6th 2022, 9:38:32 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class image_trans(object):
    def __init__(self, image_size, channel=3):
        ''' image_size [int, int], channel = 1 or 3
        '''
        self.image_size = image_size
        if channel == 3:
            trans = [
                transforms.Resize(
                    (image_size[0], image_size[1]), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        else:
            trans = [
                transforms.Resize(
                    (image_size[0], image_size[1]), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.trans = transforms.Compose(trans)

    def get_data(self, filename):
        img = Image.open(filename)
        img = self.trans(img)
        return img


class lidar_trans(object):
    """Project the LiDAR point cloud onto top-down or spherical view"""

    def __init__(self, top_size=[64, 64], sph_size=[64, 64], z_range=[-3.0, 3.0], max_dis=30, fov_range=[-25.0, 3.0]):
        """ top_size [int, int]: define top-down view image resolution
            sph_size [int, int]: define spherical view image resolution
            z_range [float, float]: define point cloud crop values on Z
            max_dis (float): maxisimum distance of cropping on XY-plane
            fov_range [float, float]: define vertical field of view

        """
        #! For top down view
        self.proj_H, self.proj_W = (int)(top_size/2)
        self.proj_Z_min, self.proj_Z_max = z_range
        # self.proj_W = (int)(top_size[1]/2)
        # self.proj_Z_max = z_range[1]

        #! For spherical view
        self.sph_H, self.sph_W = sph_size
        self.sph_down, self.sph_up = fov_range
        # self.sph_W = sph_size[1]
        # self.sph_up = fov_range[1]

        #! For activate range
        self.max_dis = max_dis

    def top_projection(self, points):
        """ Project a pointcloud into a spherical projection image.projection.
                Function takes no arguments because it can be also called externally
                if the value of the constructor was not set (in case you change your
                mind about wanting the projection)
        """

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get projections in image coords
        proj_x = scan_x/self.max_dis
        proj_y = scan_y/self.max_dis

        # scale to image size using angular resolution
        proj_x = (proj_x + 1.0)*self.proj_W                # in [0.0, 2W]
        proj_y = (proj_y + 1.0)*self.proj_H                # in [0.0, 2H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(2*self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,2W-]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(2*self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,2H-1]

        data_grid = np.zeros((2*self.proj_H, 2*self.proj_W), dtype='float64')
        data_grid[proj_y, proj_x] = scan_z

        data_norm = (data_grid - data_grid.min()) / \
            (data_grid.max() - data_grid.min())

        return data_norm

    def sph_projection(self, points):
        """ Project a pointcloud into a spherical projection image.projection.
                Function takes no arguments because it can be also called externally
                if the value of the constructor was not set (in case you change your
                mind about wanting the projection)
        """

        # laser parameters
        fov_up = self.sph_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.sph_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(points, 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)                  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        sph_H = self.sph_H
        sph_W = self.sph_W
        proj_x *= sph_W                              # in [0.0, W] 128
        proj_y *= sph_H                              # in [0.0, H] 64

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(sph_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(sph_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

        data_grid = np.zeros((sph_H, sph_W), dtype='float32')

        indices = np.argsort(depth)[::-1]
        data_grid[proj_y[indices], proj_x[indices]] = depth[indices]
        sph_norm = (data_grid - data_grid.min()) / \
            (data_grid.max() - data_grid.min())

        return sph_norm

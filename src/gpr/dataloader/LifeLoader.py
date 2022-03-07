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
    def __init__(self, dir_path, image_size=[512, 512], resolution=0.5):
        ''' image_size [int, int]: set image resolution
            resolution [float]: resolution for point cloud voxels
        '''
        super().__init__(dir_path)
        self.resolution = resolution

        # * for lidar projections
        self.lidar_trans = lidar_trans(image_size=image_size, channel=1)

        # * Obtain raw point cloud map
        pointcloud_pcd = o3d.io.read_point_cloud(dir_path+"/dense.pcd")
        ds_pcd = pointcloud_pcd.voxel_down_sample(self.resolution)
        pcd_tree = o3d.geometry.KDTreeFlann(ds_pcd)

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        return self.get_point_cloud(idx)

    def get_point_cloud(self, frame_id: int):
        '''Get the point cloud at the `frame_id` frame.
        Raise ValueError if there is no point cloud in the dataset.
        return -> o3d.geometry.PointCloud
        '''

        [k, idx, _] = pcd_tree.search_radius_vector_3d(point_trajectory, config.DATA.RADIUS)
        pcd_data = np.asarray(ds_pcd.points)[idx, :]

        pcd_data -= point_trajectory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_data)

        sph_img = projection.do_sph_projection(pcd_data)
        sph_img[sph_img < 0] = 0.
        img = o3d.geometry.Image((sph_img * 255).astype(np.uint8))
        o3d.io.write_image(dirname+"/{:04d}_sph.png".format(frame_id), img)

        pose_name = "{:04d}_pose.npy".format(frame_id)
        np.save(dirname+"/"+pose_name, point_trajectory)


        return self.lidar_trans(self.queries[frame_id])

"""
This script is a quick test for the Pittsburgh dataset.

Created Date: Friday, March 11th 2022, 3:08:28 pm
Author: Haowen Lai, Shiqi Zhao

Copyright (c) 2022 Your Company
"""

import os
import numpy as np
from tqdm import tqdm
from glob import glob
import open3d as o3d
from matplotlib import pyplot as plt

from gpr.evaluation import get_recall
from gpr.tools import HogFeature, lidar_trans

# * Load validation set, modify the folder path here
VAL_DATA_PATH = '{}/VAL'.format('/data_hdd_1/GPR/UGV/')
VAL_TRAJ = 4
SUCCESS_DIS = 3 if VAL_TRAJ in [1, 2] else 5

# * Point cloud conversion and feature extractor
# * Load your model here
lidar_to_sph = lidar_trans(
    top_size=(512, 512),
    sph_size=(512, 512),
    max_dis=40,
    fov_range=(-90, 90),
)  # for lidar projections
hog_fea = HogFeature()

# feature extraction and take val_1 as an example
#! generate database feature here
print("\033[1;34mLoading database frames\033[0m")
feature_ref = []
poses_ref = []
database_files = sorted(glob('{}/val_{}/DATABASE/*.pcd'.format(VAL_DATA_PATH, VAL_TRAJ)))
for pcd_name in tqdm(database_files):
    #* Load pcd files and preprocess them
    pcd_database = np.asarray(o3d.io.read_point_cloud(pcd_name).points)
    pose_database = np.load('{}_pose6d.npy'.format(pcd_name.split('.pcd')[0]))[:3]

    #* example code, modify here!
    sph_img = lidar_to_sph.sph_projection(pcd_database)  # get spherical projection
    sph_img = (sph_img * 255).astype(np.uint8)
    feature_ref.append(hog_fea.infer_data(sph_img))  # get HOG feature
    poses_ref.append(pose_database)  # get corresponding pose
feature_ref = np.array(feature_ref)
poses_ref = np.array(poses_ref)

#! generate query feature here
query_folders = os.listdir('{}/val_{}/QUERY/'.format(VAL_DATA_PATH, VAL_TRAJ))
for folder in query_folders:
    print("\033[1;34mLoading query frames {}\033[0m".format(folder))
    feature_query = []
    poses_query = []
    query_files = sorted(glob('{}/val_{}/QUERY/{}/*.pcd'.format(VAL_DATA_PATH, VAL_TRAJ, folder)))
    for pcd_name in tqdm(query_files):
        pcd_query = np.asarray(o3d.io.read_point_cloud(pcd_name).points)
        pose_query = np.load('{}_pose6d.npy'.format(pcd_name.split('.pcd')[0]))[:3]

        #* example code, modify here!
        sph_img = lidar_to_sph.sph_projection(pcd_query)  # get spherical projection
        sph_img = (sph_img * 255).astype(np.uint8)
        feature_query.append(hog_fea.infer_data(sph_img))  # get HOG feature
        poses_query.append(pose_query)

    # evaluate recall
    feature_query = np.array(feature_query)
    poses_query = np.array(poses_query)
    topN_recall, one_percent_recall = get_recall(
        feature_ref, feature_query, true_threshold=1, num_neighbors=20,
        reference_poses=poses_ref, queries_poses=poses_query, success_dis=SUCCESS_DIS
    )

    # plot result
    fig, ax = plt.subplots(1, 1, dpi=200)
    plt.rcParams['font.size'] = '12'
    plt.plot(np.arange(1, len(topN_recall) + 1), topN_recall)
    plt.xticks(np.arange(1, len(topN_recall), 2))
    plt.xlabel('Top N')
    plt.ylabel('Recall %')
    plt.title('Place Recognition Analysis Traj{}_{}'.format(VAL_TRAJ, folder))
    plt.savefig('Top_N_traj{}_{}.png'.format(VAL_TRAJ, folder))

    print("\033[1;32mTraj {} {} Top 1 recall {:.2%}\033[0m".format(VAL_TRAJ, folder, topN_recall[0]))
    print("\033[1;32mTraj {} {} Top 5 recall {:.2%}\033[0m".format(VAL_TRAJ, folder, topN_recall[4]))

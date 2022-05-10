"""
This script is a quick test for the Pittsburgh dataset.

Created Date: Friday, March 11th 2022, 3:08:28 pm
Author: Haowen Lai, Shiqi Zhao

Copyright (c) 2022 Your Company
"""

from ctypes import pointer
import numpy as np
from tqdm import tqdm
from glob import glob
import open3d as o3d
from matplotlib import pyplot as plt

from gpr.evaluation import get_recall
from gpr.tools import HogFeature, lidar_trans

# * Load validation set, modify the folder path here
VAL_DATA_PATH = '{}/VAL'.format('/data_hdd_1/GPR/')

# * Point cloud conversion and feature extractor
# * Load your model here
lidar_to_sph = lidar_trans(
    top_size=(512, 512),
    sph_size=(512, 512),
    max_dis=50,
    fov_range=(-90, 90),
)  # for lidar projections
hog_fea = HogFeature()

# feature extraction and take val_1 as an example
pcd_files = sorted(glob('{}/DATABASE/val_1/*.pcd'.format(VAL_DATA_PATH)))
feature_ref = []
feature_test = []
for pcd_name in tqdm(pcd_files):
    #* Load pcd files and preprocess them
    pcd_database = np.asarray(o3d.io.read_point_cloud(pcd_name).points)
    pcd_query = np.asarray(o3d.io.read_point_cloud('{}/QUERY/val_1/{}'.format(VAL_DATA_PATH, pcd_name.split('/')[-1])).points)

    #! In each val set, for same frame index in QUERY and DATABASE are set to be ground truth pairs
    #! eg: the ground truth match of VAL/QUERY/val_1/000001.pcd is VAL/DATABASE/val_1/000001.pcd
    #! generate database feature here
    #* example code, modify here!
    sph_img = lidar_to_sph.sph_projection(pcd_database)  # get spherical projection
    sph_img = (sph_img * 255).astype(np.uint8)
    feature_ref.append(hog_fea.infer_data(sph_img))  # get HOG feature

    #! generate query feature here
    #* example code, modify here!
    sph_img = lidar_to_sph.sph_projection(pcd_query)  # get spherical projection
    sph_img = (sph_img * 255).astype(np.uint8)
    feature_test.append(hog_fea.infer_data(sph_img))  # get HOG feature

# evaluate recall
# our evaluation function get_recall() is based on our data structure stated in Line 42~43
feature_ref = np.array(feature_ref)
feature_test = np.array(feature_test)
topN_recall, one_percent_recall = get_recall(
    feature_ref, feature_test, true_threshold=1, num_neighbors=20
)

# plot result
fig, ax = plt.subplots(1, 1, dpi=200)
plt.rcParams['font.size'] = '12'
plt.plot(np.arange(1, len(topN_recall) + 1), topN_recall)
plt.xticks(np.arange(1, len(topN_recall), 2))
plt.xlabel('Top N')
plt.ylabel('Recall %')
plt.title('Place Recognition Analysis')
plt.savefig('Top_N.png')

print("Top 1 recall {:.2%}".format(topN_recall[0]))
print("Top 5 recall {:.2%}".format(topN_recall[4]))

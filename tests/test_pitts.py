"""
This script is a quick test for the Pittsburgh dataset.

Created Date: Friday, March 11th 2022, 3:08:28 pm
Author: Haowen Lai

Copyright (c) 2022 Your Company
"""

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from gpr.dataloader import PittsLoader
from gpr.evaluation import get_recall
from gpr.tools import HogFeature, lidar_trans

# * Test Data Loader, change to your datafolder
pitts_loader = PittsLoader('datasets/Pitts/gpr_pitts_test')

# * Point cloud conversion and feature extractor
lidar_to_sph = lidar_trans(
    top_size=(512, 512),
    sph_size=(512, 512),
    max_dis=50,
    fov_range=(-90, 90),
)  # for lidar projections
hog_fea = HogFeature()

# feature extraction
feature_ref = []
for idx in tqdm(range(len(pitts_loader)), desc='comp. fea.'):
    pcd = pitts_loader[idx]['pcd']
    sph_img = lidar_to_sph.sph_projection(pcd)  # get spherical projection
    sph_img = (sph_img * 255).astype(np.uint8)

    feature_ref.append(hog_fea.infer_data(sph_img))  # get HOG feature

# evaluate recall
feature_ref = np.array(feature_ref)
topN_recall, one_percent_recall = get_recall(
    feature_ref, feature_ref, true_threshold=1, num_neighbors=8
)

# plot result
fig, ax = plt.subplots(1, 1, dpi=200)
plt.rcParams['font.size'] = '12'
plt.plot(np.arange(1, len(topN_recall) + 1), topN_recall)
plt.xlabel('Top N')
plt.ylabel('Recall %')
plt.title('Place Recognition Analysis')
plt.savefig('PR_analysis.png')

print("Top 1 recall {:.2%}".format(topN_recall[0]))
print("Top 5 recall {:.2%}".format(topN_recall[4]))

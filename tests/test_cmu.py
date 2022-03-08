'''
Filename: /home/maxtom/codespace/GPR_Competition/tests/test_cmu.py
Path: /home/maxtom/codespace/GPR_Competition/tests
Created Date: Friday, March 4th 2022, 5:08:28 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

from gpr import lifeloader
from gpr.tools import Feature, to_image
from gpr.evaluation import get_recall
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

#* Test Data Loader
loader1 = lifeloader('/home/maxtom/codespace/GPR_Competition/datasets/lifelong/day_forward_1') # Change to your datafolder
loader2 = lifeloader('/home/maxtom/codespace/GPR_Competition/datasets/lifelong/day_forward_2') # Change to your datafolder
F = Feature()

feature_ref = []
feature_test = []
for idx in tqdm(range(100), total=100):
    data1 = loader1.__getitem__(10) #{'img': img, 'pcd': pcd, 'sph': sph, 'top': top}
    data2 = loader2.__getitem__(10) #{'img': img, 'pcd': pcd, 'sph': sph, 'top': top}

    #* Feature Extraction and Evaluation
    feature_ref.append(F.infer_data(to_image(data1['img'])))
    feature_test.append(F.infer_data(to_image(data2['img'])))

feature_ref = np.array(feature_ref)
feature_test = np.array(feature_test)

topN_recall, one_percent_recall = get_recall(feature_ref, feature_test)

f,ax = plt.subplots(1,1, dpi=200)
plt.rcParams['font.size'] = '12'
plt.plot(topN_recall)
plt.xlabel('Top N')
plt.ylabel('Recall %')
plt.title('Place Recognition Analysis')
plt.savefig('PR_analysis.png')

print("Top 1 recall {}".format(topN_recall[0]))
print("Top 5 recall {}".format(topN_recall[4]))
'''
Filename: /home/maxtom/codespace/GPR_Competition/tests/test_cmu.py
Path: /home/maxtom/codespace/GPR_Competition/tests
Created Date: Friday, March 4th 2022, 5:08:28 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

import sys
from gpr import lifeloader
from PIL import Image
import numpy as np
from gpr.tools import to_image

#* Test Data Loader
loader = lifeloader('/home/maxtom/codespace/GPR_Competition/datasets/lifelong/day_back_1')
data = loader.__getitem__(10) #{'img': img, 'pcd': pcd, 'sph': sph, 'top': top}

#* Test Raw Data Visualization
to_image(data['img'])
to_image(data['sph'])
to_image(data['top'])

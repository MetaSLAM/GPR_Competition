'''
Filename: /home/maxtom/codespace/GPR_Competition/tests/test_cmu.py
Path: /home/maxtom/codespace/GPR_Competition/tests
Created Date: Friday, March 4th 2022, 5:08:28 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

import sys
from gpr import lifeloader
import numpy as np

#* Test Data Loader
loader = lifeloader('/home/maxtom/codespace/GPR_Competition/datasets/lifelong/day_back_1')
loader.__getitem__(10)

#* Test Raw Data Visualization

#* Test Feature Extraction

#* Test Place Recognition Results
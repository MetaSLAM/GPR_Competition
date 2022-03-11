'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/tools/feature.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/tools
Created Date: Monday, March 7th 2022, 7:00:51 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
import numpy as np
import cv2


class HogFeature:
    def __init__(
        self,
        winSize=(512, 512),
        blockSize=(16, 16),
        blockStride=(8, 8),
        cellSize=(16, 16),
        nbins=9,
    ):
        self.winSize = winSize
        self.hog = cv2.HOGDescriptor(
            self.winSize, blockSize, blockStride, cellSize, nbins
        )

    def infer_data(self, query):
        query_desc = self.hog.compute(cv2.resize(np.array(query), self.winSize))
        return query_desc.reshape(-1)

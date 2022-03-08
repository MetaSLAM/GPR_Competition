'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/tools/feature.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/tools
Created Date: Monday, March 7th 2022, 7:00:51 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
import numpy as np
import cv2

class Feature(object):
    def __init__(self):
        print("Define your descriptor here")

    def infer_data(self, query):
        winSize = (512,512)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (16,16)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
        query_desc=hog.compute(cv2.resize(np.array(query), winSize))
        
        return query_desc.reshape(-1)
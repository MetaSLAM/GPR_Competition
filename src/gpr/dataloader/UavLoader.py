'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader/base_loader.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/dataloader
Created Date: Sunday, March 6th 2022, 9:26:53 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''
from .BaseLoader import BaseLoader
from ..tools import image_trans

class UavLoader(BaseLoader):
    def __init__(self, dir_path, image_size=[512, 512], random_rotation=False):
        ''' image_size [int, int]: set image resolution
            random_rotation: True/False, set viewpoint rotation 
        '''
        super().__init__(dir_path)

        #* for raw RGB image
        self.image_trans = image_trans(image_size=image_size, channel=3)

    def __getitem__(self, idx: int):
        '''Return the query data (Image, LiDAR, etc)'''
        return self.get_image(idx)

    def get_image(self, frame_id: int):
        '''Get the image at the `frame_id` frame.
        Raise ValueError if there is no image in the dataset.
        return -> Image.Image
        '''
        return self.image_trans(self.queries[frame_id])
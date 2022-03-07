'''
Filename: /home/maxtom/codespace/GPR_Competition/src/gpr/tools/geometry.py
Path: /home/maxtom/codespace/GPR_Competition/src/gpr/tools
Created Date: Sunday, March 6th 2022, 9:38:32 pm
Author: maxtom

Copyright (c) 2022 Your Company
'''

from PIL import Image
import torchvision.transforms as transforms

class image_trans(object):
    def __init__(self, image_size, channel=3):
        ''' image_size [int, int], channel = 1 or 3
        '''
        self.image_size = image_size
        if channel==3:
            trans = [
                transforms.Resize((image_size[0], image_size[1]), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        else:
            trans = [
                transforms.Resize((image_size[0], image_size[1]), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.trans = transforms.Compose(trans)

    def get_data(self, filename):
        img = Image.open(filename)
        img = self.trans(img)
        return img
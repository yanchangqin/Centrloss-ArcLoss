import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from classify_loss import Cnn_Net
import classify_loss
import PIL.Image as img

img_path = r'F:\ycq\centerloss\3.jpg'


class Detector():
    def __init__(self,param=r'F:\ycq\centerloss\adm-sum\100\param_net.pt',iscuda=True):
        self.param = param
        self.iscuda =iscuda
        self.cnet =classify_loss.Cnn_Net
        self.cnet.load_state_dict(torch.load(self.param))

    def detect(self,img):
        arr = np.array(img)
        print(arr)

if __name__ == '__main__':
    detector = Detector()
    with img.open(img_path) as im:
        detector.detect(im)
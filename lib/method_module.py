import torch
import numpy as np
from lib.utils import build_model, build_pebal_model, download_checkpoint
from torchvision import transforms
import skimage
import os
from pathlib import Path
import wget
import torch.nn.functional as F

class Max_logit:
    def __init__(self, backbone = 'WideResNet38', weight_path = None, class_num = 19):
        self.backbone = backbone
        self.model = build_model(backbone=backbone)

    def getscore_from_logit(self, logit):
        confidence_score, prediction = torch.max(logit, axis=1)
        anomaly_score = 1 - confidence_score
        return anomaly_score

    def anomaly_score(self, image, ret_logit = False):
        logit = self.model(image)
        anomaly_score = self.getscore_from_logit(logit)

        if ret_logit:
            return  anomaly_score, logit
        return anomaly_score

class Energy:
    def __init__(self, backbone = 'WideResNet38', weight_path = None, class_num = 19):
        self.backbone = backbone
        self.model = build_model(backbone=backbone, weight_path = weight_path)

    def getscore_from_logit(self, logit):
        anomaly_score = -(1. * torch.logsumexp(logit, dim=1))
        del logit
        return anomaly_score

    def anomaly_score(self, image, ret_logit=False):
        logit = self.model(image)
        anomaly_score = self.getscore_from_logit(logit)
        if ret_logit:
            return anomaly_score, logit
        return anomaly_score

"""
Reimplementation for PEBAL (ECCV 2021)
"""
class PEBAL:
    def __init__(self, backbone = 'WideResNet38',  weight_path = None, class_num = 19,):
        self.model = build_pebal_model(backbone = backbone,  class_num = class_num+1)
        self.class_num = class_num
        self.gaussian_smoothing = transforms.GaussianBlur(7, sigma=1)

    def getscore_from_logit(self, logit):
        in_logit = logit[:, :self.class_num]
        anomaly_score = -(1. * torch.logsumexp(in_logit, dim=1))
        anomaly_score = self.gaussian_smoothing(anomaly_score)
        return anomaly_score

    def anomaly_score(self, image, ret_logit = False):
        logit = self.model(image)

        anomaly_score = self.getscore_from_logit(logit)
        anomaly_score = self.gaussian_smoothing(anomaly_score)

        in_logit = logit[:, :self.class_num]

        if ret_logit:
            return  anomaly_score, in_logit

        return anomaly_score



import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from pathlib import Path
import wget
from lib.network.deepv3 import DeepWV3Plus, DeepR101V3PlusD_OS8_v2


def random_init(seed=0):
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class RunningMeter(object):
    def __init__(self):
        self.running_loss = []

    def update(self, loss):
        self.running_loss.append(loss.detach())

    def get_metric(self):
        avg = 0
        for p in self.running_loss:
            avg += p
        loss_avg = avg*1.0 / len(self.running_loss) if len(self.running_loss)!=0 else None
        return loss_avg

    def reset(self):
        self.running_loss = []

class Accuracy():
    def __init__(self, reduction="mean"):
        self.reduction = reduction
    def __call__(self, preds_onehot, targets_onehot):

        preds_onehot = torch.clone(preds_onehot)
        targets_onehot = torch.clone(targets_onehot)
        assert(preds_onehot.shape == targets_onehot.shape)

        numerator = preds_onehot == targets_onehot
        denominator = targets_onehot == targets_onehot

        if self.reduction == "mean":
            acc = torch.sum(numerator)*1.0 / torch.sum(denominator)
        else:
            print("Not implement yet")
            assert (0==1)

        return acc

def download_checkpoint(url, save_dir):
    print("Download PyTorch checkpoint")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = wget.download(url, out=str(save_dir))
    return filename

def build_model(backbone, class_num=19, parallel=True, weight_path=None):

    # Choose model and weight path.
    if backbone == 'WideResNet38':
        model = DeepWV3Plus(class_num)
        if not weight_path:
            weight_path = '../pretrained_model/DeepLabV3+_WideResNet38_baseline.pth'
        # Use data parallel by default.
        if parallel:
            model = nn.DataParallel(model)
        # Load state dict.
        state_dict = torch.load(weight_path)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        error_message = model.load_state_dict(state_dict, strict=False)
        print(error_message)
        model = model.cuda()
        return model

    elif backbone == 'ResNet101':
        model = DeepR101V3PlusD_OS8_v2(class_num, None)
        if not weight_path:
            weight_path = '../pretrained_model/r101_os8_base_cty.pth'
        # Use data parallel by default.
        if parallel:
            model = nn.DataParallel(model)
        # Load state dict.
        state_dict = torch.load(weight_path)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        error_message = model.load_state_dict(state_dict, strict=False)
        print(error_message)
        model = model.cuda()
        return model
    else:
        return None

def build_pebal_model(backbone, class_num=19, parallel=True, weight_path=None):
    weight_path = '../PEBAL/ckpts/pebal/best_ad_ckpt.pth'

    # Choose model and weight path.
    if backbone == 'WideResNet38':
        model = DeepWV3Plus(class_num)
    else:
        model = None

    # Load state dict.
    state_dict = torch.load(weight_path)
    state_dict = state_dict['model']

    # Remove "branck" in the name.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.split('branch1.')[-1]
        new_state_dict[name] = v

    error_message =  model.load_state_dict(new_state_dict, strict=False)
    print(error_message)

    # Use data parallel by default.
    if parallel:
        model = nn.DataParallel(model)

    model = model.cuda()

    return model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

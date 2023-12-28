import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils.FMGMM import FixedMeanGMM
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d

"""
Entropy Loss used by Tent.
"""
class EntropyLoss:
    def __init__(self, param_dict):
        self.temperature = param_dict['temperature']
    def __call__(self, anomaly_score, logit,  ret_selmap=False, ret_seperate=False, target = None, step = 0):
        softmax = torch.softmax(logit / self.temperature, dim=1)
        entropy = -torch.sum(softmax * torch.log(softmax), dim=1)
        average_entropy = torch.mean(entropy)
        return average_entropy

"""
The Proposed Anomaly-aware Self-Training (AST) Loss.
"""
class ASTLoss:
    def __init__(self, param_dict):
        self.tau1 = param_dict.get('tau1', 0.0)
        self.tau2 = param_dict.get('tau2', 1.0)
        self.sample_ratio = param_dict.get('sample_ratio', 1)

    def __call__(self, anomaly_score, logit=None, eps = 1e-8):

        # Calibrate the OOD scores into the outlier probability.
        ax, bx = self.get_calibration_params(anomaly_score)
        anomaly_prob = torch.sigmoid((anomaly_score - ax) / bx)

        # Select a subset of pixels with high confidence.
        pseudo_labels = (anomaly_prob > 0.5).detach().long()
        pseudo_labels[(anomaly_prob > self.tau1)&(anomaly_prob < self.tau2)] = -1

        # Introduce the class weights to cope with the class imbalance between inliers and outliers.
        if (pseudo_labels==1).sum() == 0 or (pseudo_labels==0).sum() == 0:
            weight_ratio = 1
        else:
            weight_ratio =  (pseudo_labels==0).sum() / (pseudo_labels==1).sum()

        # The final loss. Correspond to Eq. (10) in the paper.
        g = anomaly_prob.unsqueeze(1)
        f = torch.softmax(logit, dim=1)
        t = pseudo_labels.float().unsqueeze(1)
        loss = -((((f*(1-t)*torch.log(f*(1-g)+eps)).sum(1) + weight_ratio * t*torch.log(g+eps)))[t!=-1]).mean()

        if np.isnan(loss.item()):
            loss = anomaly_score.sum() * 0 # invalid loss, no grad.

        return loss

    def get_calibration_params(self, score, MAX_NUM = 1000, BIN_NUM = 200):
        if isinstance(score, torch.Tensor):
            all_data = score.detach().cpu().numpy()
        else:
            all_data = score

        #  We sample 1% of the total data in clustering to reduce computation overhead.
        if len(all_data.flatten()) > MAX_NUM and self.sample_ratio < 1:
            selected_data = np.random.choice(all_data.flatten(), int(len(all_data.flatten()) * self.sample_ratio))
        else:
            selected_data = all_data.flatten()

        # We employ a peak-finding algorithm to identify the right-most peak in the distribution of OOD score.
        # The goal is to mitigate potential issues arising from the presence of multiple peaks in the inlier distribution.
        h = np.histogram(all_data, bins=BIN_NUM)
        h_smooth = gaussian_filter1d(h[0], sigma=1)
        peaks = argrelextrema(h_smooth, np.greater, mode = 'wrap')
        right_peak = h[1][peaks[0][-1]]

        model = FixedMeanGMM(fixed_mean=right_peak, n_components=2)
        model.fit(selected_data.reshape(-1, 1))

        # Set the calibration parameter a(x) as the value achieving equal probability under two Gaussian distributions
        uniform_points = np.linspace(selected_data.min(), selected_data.max(), num=1000).reshape(-1, 1)
        probs = model.predict_proba(uniform_points)
        diffs = np.abs(probs[:, 0] - probs[:, 1])
        equal_prob_index = np.where(diffs == diffs.min())[-1][-1]
        ax = uniform_points[equal_prob_index][0]

        # Set b(x) as the standard derivation.
        bx = all_data.std()

        return ax, bx
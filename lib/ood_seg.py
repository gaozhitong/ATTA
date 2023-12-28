import os
import time
import datetime
import h5py
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support
from lib.dataset.data_set import *
from lib.configs.parse_arg import opt, args
from lib.network.mynn import Upsample
from lib.network.deepv3 import *
from lib.utils import *
from lib.utils.metric import *
from lib.utils.img_utils import Compose, Normalize, ToTensor, Resize, Distortion, ResizeImg, Fog, ColorJitter, GaussianBlur
import lib.loss as loss

class OOD_Model(object):
    def __init__(self, method):
        super(OOD_Model, self).__init__()
        self.since = time.time()
        self.log_init()
        self.phases = []

        if not args.trans_type:
            trans_type = opt.data.trans_type
        else:
            trans_type = args.trans_type

        self.data_loaders ={
            'test': DataLoader(self.build_dataset(trans_type=trans_type), batch_size=opt.train.test_batch, drop_last=False,
                                num_workers=opt.data.num_workers, shuffle=False, pin_memory=True)}
        self.method = method
        self.best = {}

    """
    Prepare dataloader, params, optimizer, loss etc.
    """
    def build_dataset(self, dataset = None, trans_type = 'test'):
        # data transformation
        m_transforms = {
            # Testing Transformation
            'test': Compose([
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            # Transformation for adding Domain-Shift
            'distortion': Compose([
                Distortion(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            'color_jitter':Compose([
                ColorJitter(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            'gaussian_blur': Compose([
                GaussianBlur(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
            'fog': Compose([
                Fog(),
                ToTensor(),
                Normalize(mean=opt.data.mean, std=opt.data.std),
            ]),
        }

        if not dataset:
            dataset = args.dataset

        # Inlier dataset
        if dataset == "Cityscapes_train":
            ds = Cityscapes(split="train", transform=m_transforms[trans_type])
        elif dataset == "Cityscapes_val":
            ds = Cityscapes(split="val", transform=m_transforms[trans_type])

        # Anomaly dataset
        elif dataset == "RoadAnomaly":
            ds = RoadAnomaly(transform=m_transforms['test'])
        elif dataset == "FS_LostAndFound":
            ds = Fishyscapes(split="LostAndFound", transform=m_transforms['test'])
        elif dataset == "FS_Static":
            ds = Fishyscapes(split="Static", transform=m_transforms['test'])
        elif dataset == "RoadAnomaly21":
            ds = RoadAnomaly21(transform=m_transforms['test'])
        elif dataset == "RoadObstacle21":
            ds = RoadObstacle21(transform=m_transforms['test'])

        # Constructed Dataset with Domain-Shift
        elif dataset == "FS_Static_C":
            ds = Fishyscapes(split="Static", transform=m_transforms['distortion'], domain_shift=False)
        elif dataset == "FS_Static_C_sep":
            ds = Fishyscapes(split="Static", transform=m_transforms[trans_type], domain_shift=False)
        else:
            self.logger.warning("No dataset!")

        print(ds, len(ds))
        return ds

    def configure_trainable_params(self, trainable_params_name):
        self.method.model.requires_grad_(False)
        params = []
        names = []
        for nm, m in self.method.model.named_modules():
            if (trainable_params_name == 'bn' and isinstance(m, nn.BatchNorm2d))\
            or (trainable_params_name != 'bn' and trainable_params_name in nm):
                for np, p in m.named_parameters():
                    if f"{nm}.{np}" not in names:
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def set_sbn_momentum(self, momentum = None):
        for name, module in self.method.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = momentum

    def configure_bn(self, momentum = 0.0):
        for nm, m in self.method.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if opt.train.instance_BN:
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                else:
                    m.adapt = True
                    m.momentum = momentum
        return

    def build_optimizer(self, params, lr):
        if opt.train.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay = opt.train.weight_decay)
        elif opt.train.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=opt.train.momentum)

        return optimizer

    def build_loss(self):
        Criterion = getattr(loss, opt.loss.name)
        criterion = Criterion(opt.loss.params)
        return criterion

    def tta_preparation(self):
        if opt.model.trainable_params_name is not None:
            params, names = self.configure_trainable_params(opt.model.trainable_params_name)
            self.optimizer = self.build_optimizer(params, opt.train.lr)
            self.criterion = self.build_loss()

        if opt.train.episodic: # Store Initial Model and Optimizer Stat.
            self.initial_model_state = deepcopy(self.method.model.state_dict())
            self.initial_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.configure_bn()

        return


    """
    Main functions for test time adaptation.
    """

    def atta(self, img, ret_logit =False):
        # We use Episodic Training Manner by default.
        if opt.train.episodic:
            self.method.model.load_state_dict(self.initial_model_state)
            self.optimizer.load_state_dict(self.initial_optimizer_state)
            self.configure_bn()

        # 1. Selective Bacth Normalization
        with torch.no_grad():
            ds_prob = self.get_domainshift_prob(img)
            self.set_sbn_momentum(momentum=ds_prob)

        # 2. Anomaly-aware Self-Training
        anomaly_score, logit = self.method.anomaly_score(img, ret_logit=True)
        loss = self.criterion(anomaly_score, logit)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Forward to get the final output.
        with torch.no_grad():
            # Since only the final block params are updated, we only need to recalculate for this block.
            feature = self.method.model.module.final(self.method.model.module.dec0)
            logit = Upsample(feature, img.size()[2:])
            anomaly_score = self.method.getscore_from_logit(logit)

        if ret_logit:
            return anomaly_score, logit
        return anomaly_score

    def tent(self, img, ret_logit =False):
        anomaly_score, logit = self.method.anomaly_score(img, ret_logit=True)
        loss = self.criterion(anomaly_score, logit)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            anomaly_score, logit = self.method.anomaly_score(img, ret_logit=True)
        if ret_logit:
            return anomaly_score, logit
        return anomaly_score

    """
    Inference function for detect unknown classes.
    """

    def inference(self):
        self.method.model.train(False)
        if opt.train.tta is not None:
            self.tta_preparation()
        anomaly_score_list = []
        ood_gts_list = []

        for (i,data) in tqdm(enumerate(self.data_loaders['test'])):
            img, target = data[0].cuda(), data[1].numpy()
            if opt.train.tta == 'atta':
                anomaly_score = self.atta(img)
            elif opt.train.tta == 'tent':
                anomaly_score = self.tent(img)
            else:
                anomaly_score = self.method.anomaly_score(img)
            anomaly_npy = anomaly_score.detach().cpu().numpy()
            #self.calculate_metrcis(target, anomaly_npy, i) # Uncomment this for debuggging.
            ood_gts_list.append(target)
            anomaly_score_list.append(anomaly_npy)

        roc_auc, prc_auc, fpr95 = eval_ood_measure(np.array(anomaly_score_list), np.array(ood_gts_list))
        logging.warning(f'AUROC score for {args.dataset}: {roc_auc:.2%}')
        logging.warning(f'AUPRC score for {args.dataset}: {prc_auc:.2%}')
        logging.warning(f'FPR@TPR95 for {args.dataset}: {fpr95:.2%}')

    """
    Inference function for known class evaluation measures.
    """

    def inference_known(self):
        self.method.model.train(False)
        all_results = []
        with torch.no_grad():
            for (i,data) in tqdm(enumerate(self.data_loaders['test'])):
                img, ood_gts, target = data[0].cuda(), data[1].long(), data[2].long()
                outputs = self.method.anomaly_score(img, ret_logit = True)[1]
                pred = outputs.argmax(1).detach().cpu().numpy()
                ood_gts = ood_gts.numpy()
                target = target.numpy()

                label_inlier = target[(ood_gts == 0) & (target != 255)]
                pred_inlier = pred[(ood_gts == 0) & (target != 255)]

                hist_tmp, labeled_tmp, correct_tmp = hist_info(19, pred_inlier, label_inlier)
                results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}
                all_results.append(results_dict)
        m_iou, m_acc = compute_metric(all_results)
        logging.warning("current mIoU is {}, mAcc is {}".format(m_iou, m_acc))
        return


    """
    Functions related to BN-based domain shift detection.
    """

    def get_domainshift_prob(self, x, threshold = 50.0, beta = 0.1, epsilon = 1e-8):
        # Perform forward propagation
        self.method.anomaly_score(x)

        # Calculate the aggregated discrepancy
        discrepancy = 0
        for i, layer in enumerate(self.method.model.modules()):
            if isinstance(layer, nn.BatchNorm2d):
                mu_x, var_x = layer.mean, layer.var
                mu, var = layer.running_mean, layer.running_var
                # Calculate KL divergence
                discrepancy = discrepancy + 0.5 * (torch.log((var + epsilon) / (var_x + epsilon)) + (var_x + (mu_x - mu) ** 2) / (
                        var + epsilon) - 1).sum().item()

        # Training Data Stat. (Use function 'save_bn_stats' to obtain for different models).
        if opt.model.backbone == 'WideResNet38':
            train_stat_mean = 825.3230302274227
            train_stat_std = 131.76657988963967
        elif opt.model.backbone == 'ResNet101':
            train_stat_mean = 2428.9796256740888
            train_stat_std = 462.1095033939578

        # Normalize KL Divergence to a probability.
        normalized_kl_divergence_values = (discrepancy - train_stat_mean) / train_stat_std
        momentum = sigmoid(beta * (normalized_kl_divergence_values - threshold))
        return momentum

    def save_bn_stats(self):
        self.method.model.train(False)
        stats_list = []

        with torch.no_grad():
            for data in tqdm(self.data_loaders['test']):
                img, target = data[0].cuda(), data[1].cuda().long()
                self.method.anomaly_score(img)
                discrepancy = 0
                for i, layer in enumerate(self.method.model.modules()):
                    if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.LayerNorm):
                        mu_x, var_x = layer.mean, layer.var
                        mu, var = layer.running_mean, layer.running_var
                        # Calculate KL divergence
                        discrepancy += 0.5 * (torch.log((var + epsilon) / (var_x + epsilon)) +
                                              (var_x + (mu_x - mu) ** 2) / (var + epsilon) - 1).sum()
                stats_list.append(discrepancy.item())

        stats = np.array(stats_list)

        print(f'Saving stats/{args.dataset}_{opt.model.backbone}_{opt.model.method}_stats.npy')
        np.save(f'stats/{args.dataset}_{opt.model.backbone}_{opt.model.method}_stats.npy', stats)

        return

    """
    Functions related to logging (for debug usage).
    """

    def log_init(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        logfile = opt.log_dir + "/log.txt"
        fh = logging.FileHandler(logfile)#, mode='w') # whether to clean previous file
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.handlers = [fh, ch]

        logging.info(str(opt))
        logging.info('Time: %s' % datetime.datetime.now())

    def log_epoch(self, data, epoch, phase='train'):
        phrase = '{} Epoch: {} '.format(phase, epoch)
        for key, value in data.items():
            phrase = phrase + '{}: {:.4f} '.format(key, value)
        logging.warning(phrase)

    def calculate_metrcis(self, target, anomaly_npy, id):
        if (target == 1).sum() > 0:
            roc_auc, prc_auc, fpr95 = eval_ood_measure(anomaly_npy, target, train_id_out=1)
            running_terms = {'roc_auc': roc_auc, 'prc_auc': prc_auc, 'fpr95': fpr95}
            self.log_epoch(running_terms, id)
        else:
            self.logger.info("The image contains no outliers.")
        return
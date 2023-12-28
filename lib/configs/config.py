import yaml
from easydict import EasyDict as edict

config = edict()

# 1. data_dir
config.data_dir = ''
config.model_dir = ''
config.log_dir = ''
config.tb_dir = ''
config.out_dir = ''
config.dataset = ''

# 2. data related
config.data = edict()
config.data.class_num = 19
config.data.in_channels = 3
config.data.num_workers = 4
config.data.mean = [0.485, 0.456, 0.406]
config.data.std = [0.229, 0.224, 0.225]
config.data.resized = False
config.data.trans_type = 'test'

# 3. model related
config.model = edict()
config.model.method = ''
config.model.finetune_model_path = ''
config.model.weight_path = None
config.model.backbone = 'WideResNet38'
config.model.trainable_params_name = None
config.model.config_bn_online = False

# 4. training params
config.train = edict()
config.train.test_batch = 1
config.train.tta = None

config.train.optimizer = 'Adam'
config.train.lr = 1e-2
config.train.momentum = 0.9
config.train.weight_decay = 1e-4

config.train.scheduler = None

config.train.domain_detector = False
config.train.instance_BN = False

config.train.episodic = False

# 5. loss related
config.loss = edict()
config.loss.name = ''
config.loss.params = {}

# update method
def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                config[k] = v


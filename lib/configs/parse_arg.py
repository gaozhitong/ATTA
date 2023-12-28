import argparse
from lib.configs.config import config, update_config

def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', default='exp/baselines/maxlogit.yaml', help='experiment configure file name', type=str)  #
    parser.add_argument('--id', default='exp01', type=str, help='Experiment ID')
    parser.add_argument('--dataset', default='RoadAnomaly', type=str, help='Experiment ID')
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--run', help='run function name', type=str, default='inference')
    parser.add_argument('--method', help='OOD Detection method name', type=str, default=None)
    parser.add_argument('--trans_type', help='Choose Added Domain Shift', type=str, default=None)
    args, rest = parser.parse_known_args()

    update_config(args.cfg)
    args = parser.parse_args()

    return args

# default complete
def default_complete(config, id):
    import os.path as osp
    project_dir = osp.abspath('.')
    if config.log_dir == '' :
        config.log_dir = osp.join(project_dir, 'outputs/' + id)
    return config

args = parse_args()
if config.model.method == '':
    config.model.method = args.method
opt = default_complete(config, args.dataset + '/' + config.model.method + '/' + args.id)
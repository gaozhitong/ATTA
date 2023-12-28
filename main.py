import os
from lib.utils import random_init
from lib.ood_seg import OOD_Model
from lib.configs.parse_arg import opt, args
import lib.method_module as method_module

if __name__ == '__main__':
    random_init(args.seed)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)

    if args.method is not None:
        opt.model.method = args.method

    method_object = getattr(method_module, opt.model.method)
    method = method_object(opt.model.backbone, opt.model.weight_path)

    ood = OOD_Model(method)
    run_fn = getattr(ood, args.run)
    run_fn()

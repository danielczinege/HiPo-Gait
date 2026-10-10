
import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for HiPo-Gait training & test.')
parser.add_argument('--local-rank', "--local_rank", type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
parser.add_argument('--cfgs', type=str,
                    default='model_cfgs/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--out_dir', type=str,
                    default='output/', help="path of output directory")
parser.add_argument('--seed', type=int, default=0,
                    help="run seed; each process uses seed * 4 + rank (at most 4 GPUs), so 0 keeps the original seeding")
parser.add_argument('--lr_swap', default=None, metavar='CONDITION:PART:RATE',
                    help="test only: swap left and right in the test data first, e.g. B:feet:0.05 (see LRSwap in data/transform.py)")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join(opt.out_dir, cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    # Distinct runs never share a process seed: seed 0 uses 0..3, seed 1 uses 4..7, ...
    # The stride is fixed at 4, not the GPU count, so a seed gives rank 0 (the initial
    # weights and the batch order) the same seed on 1, 2 or 4 GPUs.
    if torch.distributed.get_world_size() > 4:
        raise ValueError("--seed assumes at most 4 GPUs, got {}".format(torch.distributed.get_world_size()))
    seed = opt.seed * 4 + torch.distributed.get_rank()
    init_seeds(seed)


def add_lr_swap(cfgs, lr_swap):
    """Puts LRSwap in front of the test transform; lr_swap is 'CONDITION:PART:RATE'."""
    condition, part, rate = lr_swap.split(':')
    swap = {'type': 'LRSwap', 'condition': condition, 'part': part, 'rate': float(rate)}
    test_transform = cfgs['evaluator_cfg']['transform'][0]   # one per input type; here one
    cfgs['evaluator_cfg']['transform'] = [{'type': 'Compose', 'trf_cfg': [swap, test_transform]}]


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training, opt.out_dir)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    msg_mgr.log_info('Model arquitecture')
    msg_mgr.log_info(str(model))
    model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)
    if opt.lr_swap:
        assert opt.phase == 'test', '--lr_swap is for testing only'
        add_lr_swap(cfgs, opt.lr_swap)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
    torch.distributed.destroy_process_group()

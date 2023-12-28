# coding: utf-8

import argparse
import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose

from confgf import models as score_model
from confgf import dataset
from confgf import runner as score_runner
from discrimintor import model, runner, utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--score_config_path', type=str, help='path of score config', default="config/qm9_default.yml")
    parser.add_argument('--discriminator_config_path', type=str, help='path of discriminator config',
                        default="config/qm9_dg_default.yml")
    parser.add_argument('--generator', type=str, help='type of generator [ConfGF, ConfGFDist]', default="ConfGF")
    parser.add_argument('--w_dg', type=float, help='weight of discriminator', default="2.000")
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('--score_config_path', type=str, help='path of score config', required=True)
    # parser.add_argument('--discriminator_config_path', type=str, help='path of discriminator config', required=True)
    # parser.add_argument('--generator', type=str, help='type of generator [ConfGF, ConfGFDist]', required=True)
    parser.add_argument('--num_repeat', type=int, default=1, help='end idx of test generation')
    parser.add_argument('--start', type=int, default=0, help='start idx of test generation')
    parser.add_argument('--end', type=int, default=3, help='end idx of test generation')
    parser.add_argument('--smiles', type=str, default=None, help='smiles for generation')
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')
    parser.add_argument('--tag', type=str, default="", help='overwrite config seed')

    args = parser.parse_args()
    with open(args.discriminator_config_path, 'r') as f:
        dg_config = yaml.safe_load(f)
    dg_config = EasyDict(dg_config)

    with open(args.score_config_path, 'r') as f:
        score_config = yaml.safe_load(f)
    score_config = EasyDict(score_config)

    if args.seed != 2021:
        score_config.train.seed = args.seed

    if score_config.test.output_path is not None:
        score_config.test.output_path = os.path.join(score_config.test.output_path, score_config.model.name)
        if not os.path.exists(score_config.test.output_path):
            os.makedirs(score_config.test.output_path)

    # check device
    gpus = list(filter(lambda x: x is not None, score_config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)
    score_config.train.device = device
    score_config.train.gpus = gpus

    print(score_config)
    print(dg_config)

    # set random seed
    np.random.seed(score_config.train.seed)
    random.seed(score_config.train.seed)
    torch.manual_seed(score_config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(score_config.train.seed)
        torch.cuda.manual_seed_all(score_config.train.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch')

    load_path = os.path.join(score_config.data.base_path, '%s_processed' % score_config.data.dataset)
    print('loading data from %s' % load_path)

    train_data = []
    val_data = []
    test_data = []

    if args.test_set is not None:
        with open(os.path.join(load_path, args.test_set), "rb") as fin:
            test_data = pickle.load(fin)
    elif score_config.data.test_set is not None:
        with open(os.path.join(load_path, score_config.data.test_set), "rb") as fin:
            test_data = pickle.load(fin)
    else:
        raise ValueError("do you set the test data ?")

    print('train size : %d  ||  val size: %d  ||  test size: %d ' % (len(train_data), len(val_data), len(test_data)))
    print('loading data done!')

    transform = Compose([
        utils.AddHigherOrderEdges(order=score_config.model.order),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])
    train_data = dataset.GEOMDataset(data=train_data, transform=transform)
    val_data = dataset.GEOMDataset(data=val_data, transform=transform)
    test_data = dataset.GEOMDataset_PackedConf(data=test_data, transform=transform)
    print('len of test data: %d' % len(test_data))

    score_model = score_model.DistanceScoreMatch(score_config)
    discriminator_model = model.SDE(dg_config)
    # optimizer = utils.get_optimizer(config.train.optimizer, model)
    optimizer = None
    # scheduler = utils.get_scheduler(config.train.scheduler, optimizer)
    scheduler = None

    # solver = runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, gpus, config)
    #
    assert score_config.test.init_checkpoint is not None
    assert dg_config.test.init_checkpoint is not None
    # solver.load(config.test.init_checkpoint, epoch=config.test.epoch)
    solver = score_runner.DefaultRunner(train_data, val_data, test_data, score_model, optimizer, scheduler, gpus,
                                        score_config)
    solver.load(score_config.test.init_checkpoint, epoch=score_config.test.epoch)
    dg_solver = runner.DefaultRunner(train_data, val_data, test_data, score_model, discriminator_model, optimizer,
                                     scheduler, gpus, score_config, dg_config)
    dg_model = dg_solver.load_dg(dg_config.test.init_checkpoint, dg_config.test.epoch)

    dg_model = dg_model.to(args.device)
    # solver = runner.DefaultRunner(train_data, val_data, test_data, score_model, discriminator_model, optimizer,
    #                               scheduler, gpus, score_config, dg_config)
    # solver.load(score_config.test.init_checkpoint, dg_config.test.init_checkpoint,
    #             score_epoch=score_config.test.epoch, disc_epoch=dg_config.test.epoch)

    if args.smiles is not None:
        solver.generate_samples_from_smiles(args.smiles, args.generator,
                                            num_repeat=1, keep_traj=True,
                                            out_path=score_config.test.output_path)

    if args.start != -1 and args.end != -1:
        solver.dg_generate_samples_from_testset(args.start, args.end,
                                                args.generator, num_repeat=args.num_repeat, dg_model=dg_model,w_dg=args.w_dg,
                                                out_path=dg_config.test.output_path, seed=args.seed,tag=args.tag)

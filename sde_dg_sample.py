# python sde_sample.py --config_path bash_sde/drugs_ema.yml --start 0 --end 200
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
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.transforms import Compose
from tqdm import tqdm\




from discrimintor import model, runner, utils
from discrimintor.data import dataset
from models.epsnet import get_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--score_config_path', type=str, help='path of dataset', default="configs/qm9_default.yml")
    # parser.add_argument('--score_config_path', type=str, help='path of dataset', default="bash_sde/qm9_ema.yml")
    # parser.add_argument('--disc_config_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--disc_config_path', type=str, help='path of dataset', default="configs/qm9_dg_default.yml")
    # parser.add_argument('--disc_config_path', type=str, help='path of dataset', default="bash_sde/qm9_ema_discriminator.yml")
    parser.add_argument('--num_repeat', type=int, default=None, help='end idx of test generation')
    parser.add_argument('--start', type=int, default=800, help='start idx of test generation')
    parser.add_argument('--end', type=int, default=1000, help='end idx of test generation')
    parser.add_argument('--smiles', type=str, default=None, help='smiles for generation')
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')
    parser.add_argument('--w_global', type=float, default=0.3, help='overwrite config seed')

    args = parser.parse_args()
    with open(args.disc_config_path, 'r') as f:
        disc_config = yaml.safe_load(f)
    disc_config = EasyDict(disc_config)

    with open(args.score_config_path, 'r') as f:
        score_config = yaml.safe_load(f)
    score_config = EasyDict(score_config)

    if args.seed != 2021:
        disc_config.train.seed = args.seed

    if disc_config.test.output_path is not None:
        disc_config.test.output_path = os.path.join(disc_config.test.output_path, disc_config.model.name)
        if not os.path.exists(disc_config.test.output_path):
            os.makedirs(disc_config.test.output_path)

    # check device
    gpus = list(filter(lambda x: x is not None, disc_config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in disc_config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')

    print("Let's use", len(gpus), "GPUs!")
    print("Using device %s as main device" % device)
    disc_config.train.device = device
    disc_config.train.gpus = gpus

    print(disc_config)

    # set random seed
    np.random.seed(disc_config.train.seed)
    random.seed(disc_config.train.seed)
    torch.manual_seed(disc_config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(disc_config.train.seed)
        torch.cuda.manual_seed_all(disc_config.train.seed)
    torch.backends.cudnn.benchmark = True
    print('set seed for random, numpy and torch')

    load_path = os.path.join(disc_config.data.base_path, '%s_processed' % disc_config.data.dataset)
    print('loading data from %s' % load_path)

    test_data = []
    train_data = []
    val_data = []
    if disc_config.data.test_set is not None:
        with open(os.path.join(load_path, disc_config.data.test_set), "rb") as fin:
            test_data = pickle.load(fin)
    else:
        raise ValueError("do you set the test data ?")
    transform = Compose([
        utils.AddHigherOrderEdges(order=disc_config.model.order),
        utils.AddEdgeLength(),
        utils.AddPlaceHolder(),
        utils.AddEdgeName()
    ])

    test_data = dataset.GEOMDataset_PackedConf(data=test_data, transform=transform)
    print('len of test data: %d' % len(test_data))
    score_model = get_model(score_config.model).to(args.device)
    # score_model = score_model.SDE(score_config)
    discriminator_model = model.SDE(disc_config)
    optimizer = None
    scheduler = None
    solver = runner.DefaultRunner(train_data, val_data, test_data, score_model, discriminator_model, optimizer, scheduler, gpus, disc_config)
    solver.load(score_config.test.init_checkpoint, disc_config.test.init_checkpoint, score_epoch=score_config.test.epoch, disc_epoch=disc_config.test.epoch)
    if args.start != -1 and args.end != -1:
        solver.sde_generate_samples_from_testset(args.start, args.end, \
                                                 num_repeat=args.num_repeat,
                                                 out_path=disc_config.test.output_path,
                                                 file_name='sdegen_drugs_dg_gen_{}_{}_3.pkl'.format(
                                                     disc_config.test.gen.num_euler_steps, \
                                                     disc_config.test.gen.num_langevin_steps))
        pass

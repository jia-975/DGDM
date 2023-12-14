'''
nohup python train.py --config_path ./bash_sde/drugs_ema.yml > sdegen_drugs.log 2>&1 &
'''
import argparse
import copy

import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict
import torch
from discrimintor import model, data, runner, utils
from discrimintor.data import GEOMLabelDataset
from discrimintor.utils import logger
from sklearn.model_selection import StratifiedShuffleSplit


def conformations_split(gen_data):
    # 假设gen_data是一个列表，里面是对象，每个对象有pos_gen和num_node两个属性
    # pos_gen是一个(num_repeat * num_node, 3)的张量，num_node是一个标量
    # 我们想把每个对象的pos_gen按照num_node切分成num_repeat个对象，每个对象的pos_gen是(num_node, 3)的张量

    # 创建一个空列表，用来存储切分后的对象
    split_data = []

    # 遍历gen_data中的每个对象
    for obj in gen_data:
        # 获取对象的pos_gen和num_node属性
        pos_gen = obj.pos_gen
        num_node = len(obj.atom_type)

        # 计算num_repeat的值，也就是要切分成多少个对象
        num_repeat = pos_gen.size(0) // num_node

        # 按照num_node对pos_gen进行切分，得到一个长度为num_repeat的列表，每个元素是一个(num_node, 3)的张量
        pos_gen_list = torch.split(pos_gen, num_node, dim=0)

        # 遍历pos_gen_list中的每个元素
        for pos in pos_gen_list:
            # 创建一个新的对象，复制原对象的所有属性，除了pos_gen
            new_obj = copy.copy(obj)

            # 将新对象的pos_gen属性设置为当前元素
            new_obj.pos = pos

            # 将新对象添加到split_data列表中
            split_data.append(new_obj)
    return split_data
    # 最后split_data列表中就有len(gen_data) * num_repeat个对象，每个对象的pos_gen属性是(num_node, 3)的张量


def get_data(gen_dir, true_dir, num_data):
    # Prepare real data
    with open(true_dir, 'rb') as fin:
        real_data = pickle.load(fin)
    # Prepare fake data
    with open(gen_dir, 'rb') as fin:
        gen_data = pickle.load(fin)
        print(len(gen_data))

    ## Combine the fake / real
    real_data = real_data[:num_data]
    gen_data = conformations_split(gen_data)  # 将每个构象作为一个样本

    gen_data = gen_data[:num_data]
    # 遍历 list2 中的每个对象, 删除多余属性，并对生成的多个构象进行处理（每个构象单独作为一个样本）
    for obj2 in gen_data:
        # 遍历 obj2 的属性

        for attr_name in list(obj2.keys):
            # 如果属性名不在 list1 的第一个对象中，则删除该属性
            if attr_name not in real_data[0].keys:
                delattr(obj2, attr_name)

    for item in real_data:
        if hasattr(item, 'totalenergy'):
            delattr(item, 'totalenergy')
        if hasattr(item, 'boltzmannweight'):
            delattr(item, 'boltzmannweight')

    train_data = real_data + gen_data
    train_label = torch.zeros(len(train_data))
    train_label[:len(real_data)] = 1.  # 真实数据为1，生成数据为0
    return train_data, train_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    # parser.add_argument('--config_path', type=str, help='path of dataset', default="bash_sde/qm9_ema_discriminator.yml")
    parser.add_argument('--config_path', type=str, help='path of dataset', default="config/qm9_dg_default.yml")
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    config.train.save_path = os.path.join(config.train.save_path, config.scheme.framework)
    logger.configure(dir=os.path.join(config.train.save_path, config.model.name))

    if args.seed != 2022:
        config.train.seed = args.seed

    if config.train.save and config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.model.name)
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path)

    # check device
    gpus = list(filter(lambda x: x is not None, config.train.gpus))
    assert torch.cuda.device_count() >= len(gpus), 'do you set the gpus in config correctly?'
    device = torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')

    logger.log("Let's use", len(gpus), "GPUs!")
    logger.log("Using device %s as main device" % device)
    config.train.gpus = gpus

    logger.log(config)

    # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True
    logger.log('set seed for random, numpy and torch')

    load_path = os.path.join(config.data.base_path, '%s_processed' % config.data.dataset)
    logger.log('loading data from %s' % load_path)

    if config.scheme.framework == 'dsm':  # baseline
        model = model.DistanceScoreMatch(config)
    elif config.scheme.framework == 'sde':
        model = model.SDE(config)
    elif config.scheme.framework == 'time-continuous':
        model = model.ContinuousScoreMatch(config)
    elif config.scheme.framework == 'diffusion':
        model = model.DenoisingDiffusion(config)
    else:
        raise 12312

    train_data = []
    val_data = []
    test_data = []

    # 定义一个字典
    # 将字典存储为pkl文件

    try:
        # 尝试打开文件
        with open(load_path + '/discriminator_train_data.pkl', 'rb') as f:
            # 读取文件内容
            train_data = pickle.load(f)
            # 打印数据
        with open(load_path + '/discriminator_train_labels.pkl', 'rb') as f:
            # 读取文件内容
            train_label = pickle.load(f)
            # 打印数据
    except FileNotFoundError:
        gen_dir = config.train.gen_dir
        true_dir = config.train.true_dir
        num_data = config.train.num_data
        train_data, train_label = get_data(gen_dir, true_dir, num_data)
        with open(load_path + '/discriminator_train_data.pkl', "wb") as f:
            pickle.dump(train_data, f)
        with open(load_path + '/discriminator_train_labels.pkl', "wb") as f:
            pickle.dump(train_label, f)
    transform = None
    train_dataset = GEOMLabelDataset(data=train_data, label=train_label, transform=transform)

    logger.log('train size : %d  ' % (len(train_data)))
    logger.log('loading data done!')

    optimizer = utils.get_optimizer(config.train.optimizer, model)
    scheduler = utils.get_scheduler(config.train.scheduler, optimizer)

    try:
        # 尝试打开文件
        with open(load_path + '/discriminator_test_data.pkl', 'rb') as f:
            # 读取文件内容
            test_data = pickle.load(f)
            # 打印数据
        with open(load_path + '/discriminator_test_labels.pkl', 'rb') as f:
            # 读取文件内容
            test_label = pickle.load(f)
            # 打印数据
    except FileNotFoundError:
        test_data, test_label = get_data(gen_dir=config.test.gen_dir,
                                         true_dir=load_path + '/test_data_200.pkl', num_data=200)
        with open(load_path + '/discriminator_test_data.pkl', "wb") as f:
            pickle.dump(test_data, f)
        with open(load_path + '/discriminator_test_labels.pkl', "wb") as f:
            pickle.dump(test_label, f)
    # test_dataset = []
    test_dataset = GEOMLabelDataset(data=test_data, label=test_label, transform=transform)

    solver = runner.DefaultRunner(train_dataset, val_data, test_dataset, None, model, optimizer, scheduler, gpus,
                                None, config)
    # if config.train.resume_train:

    #     solver.load(config.train.resume_checkpoint, epoch=config.train.resume_epoch, load_optimizer=True,
    #                 load_scheduler=True)
    solver.train()
    # disc_checkpoint = 'checkpoints/discriminator/sde/qm9_dg_default2021/checkpoint99'
    # logger.log("Load discriminator checkpoint from %s" % disc_checkpoint)
    # state = torch.load(disc_checkpoint)
    #
    # model.load_state_dict(state["model"])
    # solver.test(model)

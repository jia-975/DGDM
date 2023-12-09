# # 导入pickle模块，用于处理pkl文件
# import os
# import pickle
# import argparse
# import pickle
# import yaml
# import torch
# from glob import glob
# from tqdm.auto import tqdm
# from easydict import EasyDict
# 
# import discrimintor
# from models.epsnet import *
# from utils.datasets import *
# from utils.datasets import PackedConformationDataset
# from utils.transforms import *
# from utils.misc import *
# from discrimintor import *
# from discrimintor import model, runner, utils
# 
# def merge_sample_pkl(folder):
#     config_path = "configs/qm9_default.yml"
#     # config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
#     with open(config_path, 'r') as f:
#         config = EasyDict(yaml.safe_load(f))
#     transforms = Compose([
#         CountNodesPerGraph(),
#         AddHigherOrderEdges(order=config.model.edge_order),  # Offline edge augmentation
#     ])
#     # if args.test_set is None:
#     test_set = PackedConformationDataset(config.dataset.test, transform=transforms)
#     test_set_selected = []
#     for i, data in enumerate(test_set):
#         if not (800 <= i < 1000): continue
#         test_set_selected.append(data)
#     # 初始化一个空列表，用于存储所有数据
#     data_list = []
#     # 遍历文件夹下的所有文件
#     i = 1
#     data_path = os.path.join(folder, 'samples_199.pkl')
# 
#     def get_mol_key(data):
#         for i, d in enumerate(test_set_selected):
#             if d.smiles == data.smiles:
#                 return i
#         return -1
#     with open(os.path.join(data_path), "rb") as f:
#         try:
#             results = pickle.load(f)
#             results.sort(key=get_mol_key)
#             save_path = os.path.join(folder, 'samples_all.pkl')
#             with open(save_path, 'wb') as f:
#                 pickle.dump(results, f)
#         except EOFError:
#             print("ERROR")
#         # data = pickle.load(f)
#     # 将数据添加到列表中
# # 定义一个函数，用于将一个文件夹下所有以sample开头的pkl文件中的数据读取出来，写到sample_all.pkl文件中
# 
# 
# 
# 
# 
# 
# 
# # 调用函数，传入文件夹路径
# merge_sample_pkl("log/geodiff/sample_2023_08_29__16_58_31")
from utils.misc import *
log_dir = os.path.dirname(os.path.dirname("log/geodiff/checkpoints/qm9_default.pt"))
output_dir = get_new_log_dir_no_time(log_dir, 'sample', '')
print(output_dir























      )
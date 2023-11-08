# coding: UTF-8
import time
import torch
from torch.nn import DataParallel
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import CustomDataset, get_time_dif
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,5,6'
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Dupa-Project')
parser.add_argument("--local_rank", default=-1)
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, xlnet')
parser.add_argument('--data',required=True,help='choose a dataset:IMDB,Yelp,Yelp5,Amazon')
args = parser.parse_args()
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端


if __name__ == '__main__':
    dataset = '/home/sy/code/DUPA-ASA/data/'+args.data  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(88)
    torch.manual_seed(88)
    torch.cuda.manual_seed_all(88)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    train_data = CustomDataset(config.train_path, config.tokenizer, config.pad_size)
    dev_data=CustomDataset(config.dev_path, config.tokenizer, config.pad_size)
    test_data=CustomDataset(config.test_path, config.tokenizer, config.pad_size)

    train_sampler=DistributedSampler(train_data)


    train_iter = DataLoader(train_data,pin_memory=True, batch_size=config.batch_size, sampler=train_sampler)
    dev_iter = DataLoader(dev_data,pin_memory=True, batch_size=config.batch_size)
    test_iter = DataLoader(test_data,pin_memory=True, batch_size=config.batch_size)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    model = x.Model(config).to(config.device)

    # load模型要在构造DDP模型之前，且只需要在master上加载就行了。
    if torch.distributed.get_rank() == 0:
        model.load_state_dict(torch.load(config.save_path))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if model_name != 'Transformer':
        init_network(model)

    print(model.parameters)
    print(config.__dict__)
    train(config, model, train_iter, dev_iter, test_iter)
    torch.cuda.empty_cache()

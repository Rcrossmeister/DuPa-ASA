# coding: UTF-8
import time
import torch
from torch.nn import DataParallel
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import os

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

parser = argparse.ArgumentParser(description='Dupa-Project')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, xlnet')
parser.add_argument('--data',required=True,help='choose a dataset:IMDB,Yelp,Yelp5,Amazon')
args = parser.parse_args()


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


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
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config)
    model = DDP(model, device_ids=[0, 1, 2, 3]).to(config.device)


    if model_name != 'Transformer':
        init_network(model)

    print(model.parameters)
    print(config.__dict__)
    train(config, model, train_iter, dev_iter, test_iter)
    torch.cuda.empty_cache()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
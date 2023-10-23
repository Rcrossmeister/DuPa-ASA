# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif


parser = argparse.ArgumentParser(description='Dupa-Project')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, xlnet')
parser.add_argument('--data',required=True,help='choose a dataset:IMDB,Yelp,Yelp5,Amazon')
args = parser.parse_args()


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
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)

    print(model.parameters)
    print(config.__dict__)
    train(config, model, train_iter, dev_iter, test_iter)
    torch.cuda.empty_cache()

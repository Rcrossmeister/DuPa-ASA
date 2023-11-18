# coding: UTF-8
import os
import time
import torch
import numpy as np
from train_eval import train, init_network
import argparse
from utils import CustomDataset, get_time_dif
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
import bert
import xlnet


parser = argparse.ArgumentParser(description='Dupa-Project')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, xlnet')
parser.add_argument('--data',required=True,help='choose a dataset:IMDB,Yelp,Yelp5,Amazon')
args = parser.parse_args()


local_rank = int(os.environ['LOCAL_RANK'])
if __name__ == '__main__':


    init_process_group(backend='nccl', init_method='env://',
                       world_size=torch.cuda.device_count())  # nccl是GPU设备上最快、最推荐的后端

    torch.cuda.set_device(int(local_rank))

    dataset = '/home/sy/code/DUPA-ASA/data/'+args.data  # 数据集

    model_name = args.model  # bert
    if model_name=="bert":
        config = bert.Config(dataset)
    else:
        config = xlnet.Config(dataset)


    np.random.seed(88)
    torch.manual_seed(88)
    torch.cuda.manual_seed_all(88)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    train_data = CustomDataset(config.train_path, config.tokenizer, config.pad_size,config.device)
    dev_data=CustomDataset(config.dev_path, config.tokenizer, config.pad_size,config.device)
    test_data=CustomDataset(config.test_path, config.tokenizer, config.pad_size,config.device)

    print(config.device)
    train_sampler=DistributedSampler(train_data)

    def Custom_collate_fn(batch):
        token_ids = torch.tensor([item['token_ids'] for item in batch])
        seq_len=torch.tensor([item['seq_len'] for item in batch])
        masks = torch.tensor([item['mask'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return token_ids,seq_len,masks,labels

    train_iter = DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler,collate_fn=Custom_collate_fn)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size,collate_fn=Custom_collate_fn)
    test_iter = DataLoader(test_data, batch_size=config.batch_size,collate_fn=Custom_collate_fn)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    if model_name=="bert":
        model = bert.Model(config).to(config.device)
    else:
        model = xlnet.Model(config).to(config.device)

    model = DDP(model, broadcast_buffers=False, find_unused_parameters=True)

    if model_name != 'Transformer':
        init_network(model)

    print(model.parameters)
    print(config.__dict__)
    train(config, model, train_iter, dev_iter, test_iter)
    torch.cuda.empty_cache()

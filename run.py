# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Project_DUPA')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--data',required=True,help='choose a dataset:IMDB,Yelp,Yelp5,Amazon')
args = parser.parse_args()

if __name__ == '__main__':

    dataset = '/home/sy/code/DUPA-ASA/data/'+args.data  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embeddings.npz'
    raw_embedding = 'raw_embeddings.npz'
    extract_embedding = 'extract_embeddings.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_name == 'BLAT-inter':
        from utils_BLAT import build_dataset, build_iterator, get_time_dif
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    if model_name == 'BLAT-inter':
        config = x.Config(dataset, raw_embedding, extract_embedding)
    else:
        config = x.Config(dataset, embedding)
    np.random.seed(88)
    torch.manual_seed(88)
    torch.cuda.manual_seed_all(88)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    if model_name == 'BLAT-inter':
        raw_vocab, extract_vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    else:
        train_data, dev_data, test_data = build_dataset(config, args.word)
    vocab = "/home/sy/code/DUPA-ASA/bert-base-uncased/vocab.txt"
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    if model_name == 'BLAT-inter':
        config.raw_n_vocab = len(raw_vocab)
        config.extract_n_vocab = len(extract_vocab)
    else:
        config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':

        init_network(model)

    print(model.parameters)
    print(config.__dict__)
    train(config, model, train_iter, dev_iter, test_iter)
    torch.cuda.empty_cache()
# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'AttCNN'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.num_filters = 256  # 卷积核数量
        self.filter_sizes = [2, 3, 4]  # 卷积核尺寸
        self.hidden_size = 128  # lstm隐藏层
        self.num_layers = 2  # lstm层数


"""Attention-Based Convolutional Neural Networks for Relation Classification"""


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(len(config.filter_sizes) * config.num_filters, config.hidden_size)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_size, 1))
        # self.w = nn.Parameter(torch.zeros(config.hidden_size))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        emb = emb.unsqueeze(1)  # [batch_size, 1, seq_len, embeding]=[128, 1, 32, 300]
        pooled_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(emb)).squeeze(3)  # [batch_size, num_filters, seq_len - filter_size + 1]
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch_size, num_filters]
            pooled_outs.append(pool_out)
        out = torch.cat(pooled_outs, 1)  # [batch_size, num_filters * len(filter_sizes)]
        out = self.dropout(out)
        out = self.fc1(out)  # [batch_size, hidden_size]
        out = self.tanh1(out)
        alpha = F.softmax(torch.matmul(out, self.w), dim=1)  # [batch_size, seq_len]
        # alpha = F.softmax(torch.matmul(out, self.w), dim=1).unsqueeze(-1)  # [batch_size, 1, 1]
        out = out * alpha  # [batch_size, hidden_size]
        out = self.fc(out)  # [batch_size, num_classes]
        return out

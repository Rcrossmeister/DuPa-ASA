# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from collections import Counter
import torch
from transformers import BertTokenizer, BertModel


MAX_VOCAB_SIZE = 10000  # 词表长度限制 52对于10000适用
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

class Config:
    def __init__(self):
        self.train_path = "train.txt"  # 修改为你的训练数据路径
        self.dev_path = "dev.txt"  # 修改为你的验证数据路径
        self.test_path = "test.txt"  # 修改为你的测试数据路径
        self.pad_size = 360  # 根据需求设置句子的最大长度

config=Config()

def build_dataset(config, use_word):
    if use_word:
        tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")  # 使用BERT的Tokenizer
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    def load_dataset(path, pad_size=360):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    raw_content, extract_content, label = lin.split('\t')
                except:
                    lin  = lin.replace('\t', ' ').rstrip()[:-1] + '\t' + lin[-1]
                    raw_content, extract_content, label = lin.split('\t')
                bert_vocab = tokenizer_bert.get_vocab()
                raw_words_line, extract_words_line = [], []
                raw_token = tokenizer_bert(raw_content, truncation=True, max_length=pad_size)
                extract_token = tokenizer_bert(extract_content, padding='max_length', truncation=True)
                # word to id
                for word in raw_token:
                    raw_words_line = tokenizer.convert_tokens_to_ids(raw_token)
                for word in extract_token:
                    extract_words_line = tokenizer.convert_tokens_to_ids(extract_token)
                contents.append((raw_words_line, extract_words_line, int(label), pad_size))  # 传递pad_size
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return train, dev, test

# 调用build_dataset函数
train, dev, test = build_dataset(config, use_word=True)



class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x_raw = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        x_extract = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x_raw, x_extract, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        word_to_id = tokenizer.encode(train_dir, add_special_tokens=True)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

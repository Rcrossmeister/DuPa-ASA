# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from collections import Counter

MAX_VOCAB_SIZE = 10000  # 词表长度限制 52对于10000适用
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

def build_vocab(file_path, tokenizer, max_size, min_freq):

    raw_vocab_counter, extract_vocab_counter = Counter(), Counter()
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            raw_content, extract_content, label = lin.split('\t')
            raw_vocab_counter.update(tokenizer(raw_content))
            extract_vocab_counter.update(tokenizer(extract_content))

        raw_vocab_list = [word for word, freq in raw_vocab_counter.most_common(max_size) if freq >= min_freq]
        raw_vocab_dic = {word: idx for idx, word in enumerate(raw_vocab_list)}
        raw_vocab_dic.update({UNK: len(raw_vocab_dic), PAD: len(raw_vocab_dic) + 1})

        extract_vocab_list = [word for word, freq in extract_vocab_counter.most_common(max_size) if freq >= min_freq]
        extract_vocab_dic = {word: idx for idx, word in enumerate(extract_vocab_list)}
        extract_vocab_dic.update({UNK: len(extract_vocab_dic), PAD: len(extract_vocab_dic) + 1})
        return raw_vocab_dic, extract_vocab_dic

def pad_tokens(token, pad_size):
    if len(token) < pad_size:
        token.extend([PAD] * (pad_size - len(token)))
    else:
        token = token[:pad_size]
    seq_len = min(len(token), pad_size)
    return token, seq_len



def build_dataset(config, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.raw_vocab_path) and os.path.exists(config.extract_vocab_path):
        raw_vocab = pkl.load(open(config.raw_vocab_path, 'rb'))
        extract_vocab = pkl.load(open(config.extract_vocab_path, 'rb'))
    else:
        raw_vocab, extract_vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=2)
        pkl.dump(raw_vocab, open(config.raw_vocab_path, 'wb'))
        pkl.dump(extract_vocab, open(config.extract_vocab_path, 'wb'))
    print(f"Raw_Vocab size: {len(raw_vocab)}")
    print(f"Extract_Vocab size: {len(extract_vocab)}")

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
                raw_words_line, extract_words_line = [], []
                raw_token = tokenizer(raw_content)
                extract_token = tokenizer(extract_content)
                if pad_size:
                    raw_token, seq_len = pad_tokens(raw_token, pad_size)
                    extract_token, seq_len = pad_tokens(extract_token, pad_size)
                # word to id
                for word in raw_token:
                    raw_words_line.append(raw_vocab.get(word, raw_vocab.get(UNK)))
                for word in extract_token:
                    extract_words_line.append(extract_vocab.get(word, extract_vocab.get(UNK)))
                contents.append((raw_words_line, extract_words_line, int(label), seq_len))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return raw_vocab, extract_vocab, train, dev, test


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
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
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

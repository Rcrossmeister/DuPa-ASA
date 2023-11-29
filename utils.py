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
from transformers import XLNetTokenizer, XLNetModel
from transformers import BertConfig, BertModel, BertTokenizer
# xlnettokenizer = XLNetTokenizer.from_pretrained("/home/yjy/xlnet-large-cased")
# xlnetmodel = XLNetModel.from_pretrained("/home/yjy/xlnet-large-cased")
bert_model = BertModel.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased")




MAX_VOCAB_SIZE = 10000  # 词表长度限制 52对于10000适用
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号



class Config:
    def __init__(self):
        self.train_path = "/home/yjy/DuPa-ASA/data/IMDB/data/train.txt"  # 修改为你的训练数据路径
        self.dev_path = "/home/yjy/DuPa-ASA/data/IMDB/data/dev.txt"  # 修改为你的验证数据路径
        self.test_path = "/home/yjy/DuPa-ASA/data/IMDB/data/test.txt"  # 修改为你的测试数据路径
        self.pad_size = 512  # 根据需求设置句子的最大长度
        self.vocab_path = "/home/yjy/DuPa-ASA/bert-base-uncased/vocab.txt"  # 词表路径
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  # 设备


def build_dataset(config, use_word):
    if use_word:
        tokenizer = bert_tokenizer  # 使用BERT的Tokenizer
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    def load_dataset(path, pad_size):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    content, label = lin.split('\t')
                except:
                    lin = lin.replace('\t', ' ').rstrip()[:-1] + '\t' + lin[-1]
                    content, label = lin.split('\t')
                words_line = []
                token = tokenizer.encode(content, add_special_tokens=True,truncation=True,max_length=360,padding=True)  # 使用BERT的tokenizer对文本进行编码
                seq_len = len(token)

                if pad_size:
                    if len(token) < pad_size:
                        token.extend([tokenizer.pad_token_id] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                contents.append((token, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


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
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

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
    # # 加载BERT模型
    # xlnettokenizer = XLNetTokenizer.from_pretrained("/home/yjy/enter/xlnet-base-uncased")
    # xlnetmodel = XLNetModel.from_pretrained("/home/yjy/enter/xlnet-base-uncased")
    #
    #
    # train_dir = "/home/yjy/DuPa-ASA-main/data/IMDB/train.txt"
    # vocab_dir = "/home/yjy/DuPa-ASA-main/data/IMDB/vocab.txt"
    # filename_trimmed_dir = "/home/yjy/DuPa-ASA-main/data/IMDB/embeddings.npz"
    #
    # if os.path.exists(vocab_dir):
    #     word_to_id = xlnettokenizer.get_vocab()
    # else:
    #     print(False)
    #     #tokenizer = lambda x: tokenizer.encode(x, add_special_tokens=False)
    #     #word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    #     #pkl.dump(word_to_id, open(vocab_dir, 'wb'))
    #
    # max_sequence_length = 512  # 指定最大的序列长度
    #
    # embeddings = []
    #
    # with open(train_dir, "r", encoding='UTF-8') as f:
    #     for i, line in tqdm(enumerate(f.readlines()), total=len(f.readlines()), desc="Processing"):
    #         lin = line.strip()
    #         if not lin:
    #             continue
    #         content, label = lin.split('\t')
    #
    #         raw_token = xlnettokenizer.tokenize(content)
    #
    #         # 截断或缩短 raw_token 以适应最大序列长度
    #         if len(raw_token) > max_sequence_length:
    #             raw_token = raw_token[:max_sequence_length]
    #
    #         with torch.no_grad():
    #             input_ids = xlnettokenizer.convert_tokens_to_ids(raw_token)
    #             # 确保输入长度为 max_sequence_length
    #             if len(input_ids) < max_sequence_length:
    #                 input_ids += [xlnettokenizer.pad_token_id] * (max_sequence_length - len(input_ids))
    #             else:
    #                 input_ids = input_ids[:max_sequence_length]
    #
    #                 # 创建 attention_mask
    #             attention_mask = [1] * len(input_ids)
    #             with torch.no_grad():
    #                 output = xlnetmodel(torch.tensor(input_ids).unsqueeze(0),
    #                                     attention_mask=torch.tensor(attention_mask).unsqueeze(0))[0].mean(
    #                     dim=1).squeeze()
    #
    #         embeddings.append((output, int(label)))
    #
    # f.close()
    #
    # embeddings = [e[0].numpy() for e in embeddings]
    # embeddings = np.array(embeddings, dtype='float32')
    # np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

    print('1')
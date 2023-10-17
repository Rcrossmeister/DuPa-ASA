# coding: UTF-8
import os
import torch
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from transformers import BertTokenizer


MAX_VOCAB_SIZE = 10000  # 词表长度限制 52对于10000适用
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_dataset(config, use_word):  #获得vocab和各个集中的词信息
    if use_word:
        tokenizer = BertTokenizer.from_pretrained("/home/sy/code/DUPA-ASA/bert-base-uncased")  # 使用BERT的Tokenizer
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
                    content, label = lin.split('\t')  #分割文本与label

                token = tokenizer.encode(content, add_special_tokens=True, truncation=True, max_length=360,
                                         padding=True)  # 使用BERT的tokenizer对文本进行编码
                seq_len = len(token)
                if pad_size:  ##保证每行文本都为同一长度
                    if len(token) < pad_size:
                        token.extend([tokenizer.pad_token_id] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                contents.append((token, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]每个元素是一个元组，包括文本内容的索引序列、标签和序列长度。
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size  #mini-batch的大小
        self.batches = batches  #是作为batch的数据们
        self.n_batches = len(batches) // batch_size  #有多少个mini-batch
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
    # 下面的目录、文件名按需更改。
    # train_dir = "./THUCNews/data/train.txt"
    # vocab_dir = "./THUCNews/data/vocab.pkl"
    # pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    # emb_dim = 300
    # filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    # if os.path.exists(vocab_dir):
    #     word_to_id = pkl.load(open(vocab_dir, 'rb'))
    # else:
    #     # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
    #     tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    #     word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    #     pkl.dump(word_to_id, open(vocab_dir, 'wb'))
    #
    # embeddings = np.random.rand(len(word_to_id), emb_dim)
    # f = open(pretrain_dir, "r", encoding='UTF-8')
    # for i, line in enumerate(f.readlines()):
    #     # if i == 0:  # 若第一行是标题，则跳过
    #     #     continue
    #     lin = line.strip().split(" ")
    #     if lin[0] in word_to_id:
    #         idx = word_to_id[lin[0]]
    #         emb = [float(x) for x in lin[1:301]]
    #         embeddings[idx] = np.asarray(emb, dtype='float32')
    # f.close()
    # np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    print('1')
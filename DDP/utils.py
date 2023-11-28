# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset,DataLoader

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


class CustomDataset(Dataset):
    def __init__(self, path, tokenizer, pad_size):
        self.tokenizer = tokenizer
        self.data = self.load_dataset(path, pad_size)

    def load_dataset(self, path, pad_size):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                parts = lin.split('\t')
                if len(parts) != 2:
                    continue  # skip this line
                content, label = parts
                token = self.tokenizer.tokenize(content)
                token = [self.tokenizer.cls_token] + token
                seq_len = len(token)
                mask = []
                token_ids = self.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))

                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                content_1={'token_ids':token_ids,'seq_len':seq_len,'mask':mask,'label':int(label)}
                contents.append(content_1)
        return contents

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

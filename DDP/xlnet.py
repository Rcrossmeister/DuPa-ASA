# coding: UTF-8
import torch
import torch.nn as nn
from transformers import XLNetModel, XLNetTokenizer
from run import local_rank


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'xlnet'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda:{}'.format(int(local_rank)) if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 256                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.xlnet_path = '/home/sy/code/DUPA-ASA/xlnet-base-uncased'
        self.tokenizer = XLNetTokenizer.from_pretrained(self.xlnet_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(config.xlnet_path)
        for param in self.xlnet.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        outputs = self.xlnet(context, attention_mask=mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # Use the CLS token
        out = self.fc(pooled)
        return out

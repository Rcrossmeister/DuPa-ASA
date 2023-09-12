import torch
import torch.nn as nn
from transformers import BertConfig
import numpy as np
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'BLAT_Att'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')   # 设备
        """
        上路的参数
        """
        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 256                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.num_attention_heads = 1                                    # 头个数
        self.initializer_range = 0.02                                   # 正态分布的方差
        self.mask = True                                                # 有没有Mask
        """
        下路的参数
        """
        self.filter_sizes = (3, 4, 5)                                   # 卷积核尺寸
        self.num_filters = 256



class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        if config.embed % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.embed, config.num_attention_heads))
        """
        上路的参数
        """
        self.attention_head_size = int(config.hidden_size * 2 / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 和hidden_size是等价的
        self.input_dim = config.hidden_size * 2

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)  # 相当于是多少份attention权重
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)
        self.att_fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.att_fc2 = nn.Linear(self.input_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        # self.upper_fc = nn.Linear(self.input_dim,config.num_filters * len(config.filter_sizes) )
        """
        下路的参数
        """
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.under_fc = nn.Linear(config.num_filters * len(config.filter_sizes), self.all_head_size)
        """
        输出层的参数        
        """
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.all_head_size * 2, config.num_classes)
        # self.tmp_fc = nn.Linear(int(self.all_head_size + config.num_filters * len(config.filter_sizes)),config.num_classes)
        # self.lager_fc = nn.Linear((config.num_filters * len(config.filter_sizes))*2, config.num_classes)
        self.tmpout1 = nn.Linear(self.all_head_size * 2 , 64)
        self.tmpout2 = nn.Linear(64, config.num_classes)
        self.w = nn.Parameter(torch.zeros(self.all_head_size * 2))
        self.w1 = nn.Parameter(torch.zeros(config.num_filters * len(config.filter_sizes) ))

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def weight_pooling(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def initialize_mask(self, x):
        unk_id = 10000
        pad_id = 10001
        init_mask = torch.ones_like(x)
        init_mask[x == unk_id] = 0
        init_mask[x == pad_id] = 0
        init_mask[x == 0] = 0
        init_mask[x == 2] = 0
        # init_mask[x == 165] = 0
        # init_mask[x == 935] = 0
        # init_mask[x == 1819] = 0
        return init_mask

    def forward(self, x):
        init_mask = self.initialize_mask(x[0])
        emb = self.embedding(x[0])

        """
        上路的计算
        """
        attention_mask = init_mask.unsqueeze(1)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states, _ = self.lstm(emb) # [batch_size, pad_len, hidden_size*2]
        hidden_states = nn.Tanh()(hidden_states)

        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size ** 0.5
        # add attention mask 将填0的部分变成负无穷
        query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1, 1,
                                                                                    self.num_attention_heads * self.attention_head_size)
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat

        query_key_score = (self.key_att(mixed_query_key_layer) / self.attention_head_size ** 0.5).transpose(1, 2)

        # add attention mask 将填0的部分变成负无穷
        query_key_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        # query = value
        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        upper_output = self.weight_pooling(weighted_value, init_mask)
        # upper_output = self.upper_fc(upper_output)
        """
        下路的计算
        """
        under_output = emb.unsqueeze(1)
        under_output = torch.cat([self.conv_and_pool(under_output, conv) for conv in self.convs], 1)
        # 下路的attention
        # under_output = under_output.unsqueeze(-2)
        # M = nn.Tanh()(under_output)
        # alpha = F.softmax(torch.matmul(M, self.w1), dim=1).unsqueeze(-1)
        # under_att_out = under_output * alpha
        # under_att_out = torch.sum(under_att_out, 1)
        # under_output = F.relu(under_att_out)

        under_output = self.under_fc(under_output)


        """
        最终的输出处理
        """
        # 分别进行dropout
        upper_output = self.dropout(upper_output)
        under_output = self.dropout(under_output)

        # output = nn.Tanh()(output)
        # attention_scores = self.attention(output)
        # attention_weights = F.softmax(attention_scores, dim=0)
        # output = attention_weights * output

        output = torch.cat([upper_output,under_output], dim=1)
        output = self.fc(output)

        # 自己Attention的通用模块
        # output = output.unsqueeze(-2)
        # M = nn.Tanh()(output)
        # alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # att_out = output * alpha
        # att_out = torch.sum(att_out ,1)
        # output = F.relu(att_out)
        # output = self.tmpout1(output)
        # output = self.tmpout2(output)

        return output

# class SelfAttention(nn.Module):
#     def __init__(self, input_size):
#         super(SelfAttention, self).__init__()
#         self.input_size = input_size
#
#         # Query, Key, Value矩阵的线性变换层
#         self.Wq = nn.Linear(input_size, input_size)
#         self.Wk = nn.Linear(input_size, input_size)
#         self.Wv = nn.Linear(input_size, input_size)
#
#         # 输出矩阵的线性变换层
#         self.Wo = nn.Linear(input_size, input_size)
#
#     def forward(self, x):
#         # 对输入进行线性变换，得到Query、Key、Value矩阵
#         Q = self.Wq(x)
#         K = self.Wk(x)
#         V = self.Wv(x)
#
#         # 计算注意力分数
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.input_size))
#
#         # 对分数进行softmax归一化
#         attention_weights = nn.functional.softmax(scores, dim=-1)
#
#         # 对Value矩阵进行加权和
#         context_vector = torch.matmul(attention_weights, V)
#
#         # 对加权和进行线性变换，得到输出矩阵
#         output = self.Wo(context_vector)
#
#         return output

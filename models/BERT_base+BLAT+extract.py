import torch.nn as nn
from transformers import BertConfig, BertModel,BertTokenizer, BertForSequenceClassification,BertForQuestionAnswering
import torch
import numpy as np
import torch.nn.functional as F
# 这里需要换成绝对路径
# 搞一个bert-base-uncased
bert_model = BertModel.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased").to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
bert_tokenizer = BertTokenizer.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased")
extract_model = BertForQuestionAnswering.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased").to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'BERT-large+BLAT+extract'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/home/yjy/DuPa-ASA/bert-base-uncased/vocab.txt'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = bert_model.embeddings.word_embeddings.weight
        self.summarization_model = BertForSequenceClassification.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased")
        self.summarization_tokenizer = BertTokenizer.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.extract_model = BertForQuestionAnswering.from_pretrained("/home/yjy/DuPa-ASA/bert-base-uncased").to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        """
        上路的参数
        """
        self.dropout = 0.2                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 64                                          # mini-batch大小
        self.pad_size = 256                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3
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
        self.filter_sizes = (4, 5, 6, 7)                                   # 卷积核尺寸
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
                            bidirectional=True, batch_first=True, dropout=config.dropout).to(self.config.device)
        self.query = nn.Linear(self.input_dim, self.all_head_size).to(self.config.device)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads).to(self.config.device) # 相当于是多少份attention权重
        self.key = nn.Linear(self.input_dim, self.all_head_size).to(self.config.device)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads).to(self.config.device)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size).to(self.config.device)

        self.att_fc1 = nn.Linear(self.input_dim, self.input_dim).to(self.config.device)
        self.att_fc2 = nn.Linear(self.input_dim, 1).to(self.config.device)
        self.softmax = nn.Softmax(dim=-1).to(self.config.device)
        self.LayerNorm = nn.LayerNorm(512, eps=1e-12).to(self.config.device)
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

        self.apply(self.init_weights)

    '''
        新增添的东西
        '''
    #
    # def bert_summarizer(self, text):
    #     # 将文本分割成句子
    #     sentences = text.split('. ')
    #     sentences = [s for s in sentences if s]
    #
    #     # 为每个句子打分
    #     scores = []
    #     for sentence in sentences:
    #         inputs = self.summarization_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True,
    #                                               max_length=512)
    #         with torch.no_grad():
    #             output = self.summarization_model(**inputs)
    #         scores.append(output.logits[0][1].item())  # 假设正类(1)是"重要的"
    #
    #     # 选择得分最高的句子作为摘要
    #     summary = sentences[scores.index(max(scores))]
    #     return summary
    #
    # def split_into_sentences(self, input_ids):
    #     # 获取句号的token_id
    #     period_token_id = self.summarization_tokenizer.convert_tokens_to_ids('.')
    #
    #     sentences_ids = []
    #     current_sentence = []
    #     for token_id in input_ids[0]:  # 假设input_ids是一个batch，我们处理第一个
    #         current_sentence.append(token_id)
    #         if token_id == period_token_id:
    #             # 结束当前句子
    #             sentences_ids.append(torch.tensor(current_sentence))
    #             current_sentence = []
    #
    #     # 处理最后一个句子（如果它不是空的）
    #     if current_sentence:
    #         sentences_ids.append(torch.tensor(current_sentence))
    #
    #     return sentences_ids

    '''
    新增添的东西
    '''
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

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
        return init_mask
    def extract_model(self, x, init_mask):
        extract_model(x, init_mask).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        print(x)
        print(x[0])

        # x = input_ids (vocab index e.g. [205, 1, 200, 123, ..., <PAD>, <PAD>])

        # init_mask = attention_mask (mask e.g. [1, 1, 1, 0, ..., 0])
        init_mask = self.initialize_mask(x[0]).to(self.config.device)
        print(init_mask)
        # 最理想的就是通过SA的bp能把extract也训好
        # 不行的话就通过现成的库给一个抽取式摘要

        # extract_outputs = model(x, attention_mask = init_mask, start_positions=start_positions,
        #                 end_positions=end_positions)
        # extract_mask = self.initialize_mask(x[0])
        """
        extract 放在这个地方
        """
        # x = input_ids (vocab index e.g. [205, 1, 200, 123, ..., <PAD>, <PAD>])
        # init_mask = attention_mask (mask e.g. [1, 1, 1, 0, ..., 0])
        # 使用BERT模型进行摘要

        summarized_ids = self.extract_model(x[0].to(self.config.device), init_mask.to(self.config.device))

        print(summarized_ids)

        # emb: fine-tune bert 的地方
        emb = self.embedding(summarized_ids).to(self.config.device)
        # emb = self.embedding(extract_outputs)
        """ 上路的计算 """
        attention_mask = init_mask.unsqueeze(1)
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states, _ = self.lstm(emb)  # [batch_size, pad_len, hidden_size*2]
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
        upper_output = self.LayerNorm(upper_output)

        """
        下路的计算
        """
        under_output = emb.unsqueeze(1)
        under_output = torch.cat([self.conv_and_pool(under_output, conv) for conv in self.convs], 1)
        under_output = self.under_fc(under_output)

        under_output = F.relu(under_output)

        """
        最终的输出处理
        """
        # 分别进行dropout
        upper_output = self.dropout(upper_output)
        under_output = self.dropout(under_output)

        output = torch.cat([upper_output, under_output], dim=1)

        output = self.dropout(output)
        output = self.fc(output)
        return output



# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        self.model_name = 'RNN'
        self.train_path = dataset + '/data/train.txt'  # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'  # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + 'ckpt'  # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.3  # 随机失活
        self.require_improvement = 10000  # 若超过10000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 7  # epoch数
        self.batch_size = 128  # batch size
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度， 若使用了预训练词向量，则维度统一
        self.hidden_size = 128  # rnn隐藏层
        self.num_layers = 2  # rnn层数，注意RNN中的层数必须大于1，dropout才会生效


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.rnn = nn.RNN(config.embed, config.hidden_size, config.num_layers,
                          batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # 将原始数据转化成密集向量表示 [batch_size, seq_len, embedding]
        out = self.embedding(x[0])
        out, hidden_ = self.rnn(out)
        # out[:, -1, :] seq_len最后时刻的输出等价 hidden_
        out = self.fc(out[:, -1, :])
        return out

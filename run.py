import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='RNN', type=str, required=True,  help='RNN or LSTM')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    # args.word 分词方式, True是词级别，默认是False
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    # build_iterator返回格式 [([词/字在词典中的位置] ,label, len(word)), ...]
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # len(vocab)="<PAD>", len(vocab) -1 ="<UNK>"
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)

    train(config, model, train_iter, dev_iter, test_iter)

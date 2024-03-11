from importlib import import_module
import torch
import pickle as pkl
import numpy as np
import time

model_name = 'RNN'
model_path = 'THUCNews/saved_dict/' + model_name + '.ckpt'
dataset = 'THUCNews'
embedding = 'embedding_SougouNews.npz'

UNK, PAD = '<UNK>', '<PAD>'
result_dic = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science', 5: 'society',
              6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}

pred_str = '冯德伦徐若隔空传情 默认其是女友'
begin = time.time()
x = import_module('models.' + model_name)
# 取出配置文件
config = x.Config(dataset, embedding)
# 加载模型
model = x.Model(config)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
# 开启评估模式
model.eval()
# 加载字典
vocab = pkl.load(open(config.vocab_path, 'rb'))


def build_vector(word, pad_size=32):
    # 切割成字符
    token = [c for c in word.strip()]
    word_vec = []
    word_len = []
    # 每个句子的切割的维度，短补长切
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(word)))
            word_len.append(len(token))
        else:
            token = token[:pad_size]
            word_len.append(pad_size)

    for char in token:
        word_vec.append(vocab.get(char, vocab.get(UNK)))
    # 每个字在字典中的index索引
    return np.array(word_vec).reshape(1, -1), np.array(word_len)


word_vec, word_len = build_vector(pred_str)

pred_data = (torch.from_numpy(word_vec), torch.from_numpy(word_len))
ret = model(pred_data)
# 取出概率最大的index
pred_ret = torch.nn.functional.softmax(ret.data, dim=1)
result_classify = result_dic.get(int(np.argmax(pred_ret.numpy())))
print(f'predict result: {result_classify}')
print(f'cost time: {time.time() - begin} s')

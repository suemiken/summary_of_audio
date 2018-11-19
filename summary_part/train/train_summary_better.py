# coding: utf-8
import sys
sys.path.append('..')
from common import config
# GPUで実行する場合は下記のコメントアウトを消去（要cupy）
# ==============================================
# config.GPU = True
# ==============================================
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_perplexity, to_gpu
from seq2seq.proposed_layer import InputLayer
from common.seq2seq import Seq2seq
import numpy as np
import matplotlib.pyplot as plt
import pickle
from format_text import format_text


def vocab_dict_cre(text):
    corpus = []
    word_to_id = {}
    id_to_word = {}
    f_dict = {}
    cache = 0

    with MeCab('-F%m,%f[0],%h') as nm:
        for n in nm.parse(text, as_nodes=True):
            node = n.feature.split(',')
            if len(node) != 3:
                continue
            if not node[0] in self.f_dict:
                word_to_id.update({node[0]:self.cache})
                id_to_word.update({cache:node[0]})
                cache += 1
                f_dict.update({node[0]:1})
            else:
                f_dict[node[0]] += 1
            corpus.append(self.all_word_to_id[node[0]])
    corpus.append(corpus)
    return corpus

def format_document(documents):
    long_text = ''
    art_text = ''
    copora = []
    for text in documents:
        art_text = format_text(text)
        copora = vocab_dict_cre(art_text)
    return copora

def read_data(first, last, Flag='train'):
    summary_documents = []
    if Flag == 'train':
        for i in range(first,last):
            f = open('../../corpora/hori_F&Q/train/hori_corpus'+ str(i) + '.txt', "r")
            text= f.read()
            f.close()
            summary_documents.append(text)
    elif Flag == 'test':
        for i in range(first,last):
            f = open('../../corpora/hori_F&Q/summary/hori_summary'+ str(i) + '.txt', "r")
            text= f.read()
            f.close()
            summary_documents.append(text)
    elif Flag == 'eval':
        for i in range(last):
            f = open('../../corpora/hori_F&Q/summary/hori_summary'+ str(i) + '.txt', "r")
            text= f.read()
            f.close()
            summary_documents.append(text)
    return summary_documents


# ハイパーパラメータの設定
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5  # Truncated BPTTの展開する時間サイズ
lr = 0.1
max_epoch = 25
max_grad = 5.0


# 正解データの読み込みとtsの作成
summary_documents = read_data(1,8, 'test')
print(summary_documents)
inputlayer = InputLayer(summary_documents, 16)
summary_corpora, _, _ = inputlayer.get_corpus()
ts_train = inputlayer.get_train_data(summary_corpora)

# 学習データの読み込みとxsの作成
input_documents = read_data(1,8, 'train')

inputlayer = InputLayer(input_documents, 16)

corpora, id_to_word, word_to_id = inputlayer.get_corpus()
vocab_size = len(word_to_id)
xs_train = inputlayer.get_train_data(corpora)

model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
# print(corpora)
trainer = Trainer(model, optimizer)
#学習するか
learn = True
if learn:
    for epoch in range(max_epoch):
        trainer.fit(xs_train, ts_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    trainer.plot('seq2seq2seq_better')



else:
    with open('seq2seq_better.pkl', 'rb') as f:
        model.param = pickle.load(f)

#テストデータで評価
text = read_data(9,9)

corpus = format_text(text)
test_xs = inputlayer.get_train_data(corpus)
# model.generate(test_xs, id_to_word[3], 30)

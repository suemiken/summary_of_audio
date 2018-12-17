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
from input_layer import InputLayer
from common.seq2seq import Seq2seq
import numpy as np
import matplotlib.pyplot as plt
import pickle
from text_form import *
from text_form.eval import eval


# ハイパーパラメータの設定
batch_size = 3
wordvec_size = 100
hidden_size = 100
time_size = 50  # Truncated BPTTの展開する時間サイズ
lr = 0.1
max_epoch = 500
max_grad = 5.0

# 学習時に使用する

total_loss = 0
loss_count = 0
ppl_list = []

# 正解データの読み込みとtsの作成
summary_documents = []
for i in range(1,8):
    f = open('../../corpora/hori_F&Q/summary/hori_summary'+ str(i) + '.txt', "r")
    text= f.read()
    f.close()
    summary_documents.append(text)

inputlayer = InputLayer(summary_documents, 16, False)
summary_corpora, _, _ = inputlayer.get_corpus()
docu_ts = inputlayer.get_train_data(summary_corpora)

# 学習データの読み込みとxsの作成
input_documents = []
for i in range(1,8):
    f = open('../../corpora/hori_F&Q/train/hori_corpus'+ str(i) + '.txt', "r")
    text= f.read()
    f.close()
    input_documents.append(text)

inputlayer = InputLayer(input_documents, 16, False)

corpora, id_to_word, word_to_id = inputlayer.get_corpus()
vocab_size = len(word_to_id)
docu_xs = inputlayer.get_train_data(corpora)

model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

#学習するか
learn = False
docu_xs = np.array(docu_xs)
docu_ts = np.array(docu_ts)

if learn:
    acc_list = []
    for epoch in range(max_epoch):
        trainer.fit(docu_xs, docu_ts, max_epoch=1,batch_size=batch_size, max_grad=max_grad)

    trainer.plot('seq2seq2')

    with open('seq2seq2.pkl', 'wb') as f:
        pickle.dump(model.params, f)

else:
    with open('seq2seq2.pkl', 'rb') as f:
        model.param = pickle.load(f)
        
# #テストデータで評価
print(eval(inputlayer, model))

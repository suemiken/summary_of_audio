# coding: utf-8
import sys
sys.path.append('..')
from common import config
# GPUで実行する場合は下記のコメントアウトを消去（要cupy）
# ==============================================
# config.GPU = True
# ==============================================
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from commonseq2seq.input_layer import InputLayer
from common.seq2seq import Seq2seq
import numpy as np
import matplotlib.pyplot as plt
import pickle
from text_form import *


# ハイパーパラメータの設定
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5  # Truncated BPTTの展開する時間サイズ
lr = 0.1
max_epoch = 100

# 学習時に使用する変数
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
optimizer = SGD()
#学習するか
learn = True
if learn:
    for (xs, ts) in zip(docu_xs, docu_ts):
        # ミニバッチの各サンプルの読み込み開始位置を計算
        jump = (len(xs) - 1) // batch_size
        xoffsets = [i * jump for i in range(batch_size)]
        jump = (len(ts) - 1) // batch_size
        toffsets = [i * jump for i in range(batch_size)]
        xdata_size = len(xs)
        tdata_size = len(ts)
        max_iters = xdata_size // (batch_size * time_size)
        time_idx = 0

        for epoch in range(max_epoch):
            for iter in range(max_iters):
                # ミニバッチの取得
                batch_x = np.zeros((batch_size, time_size), dtype='i')
                batch_t = np.zeros((batch_size, time_size), dtype='i')

                for t in range(time_size):
                    for i, (xoffset, toffset) in enumerate(zip(xoffsets, toffsets)):
                        batch_x[i, t] = xs[(xoffset + time_idx) % xdata_size]
                        batch_t[i, t] = ts[(toffset + time_idx) % tdata_size]
                    time_idx += 1

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                optimizer.update(model.params, model.grads)
                total_loss += loss
                loss_count += 1

            # エポックごとにパープレキシティの評価
            ppl = np.exp(total_loss / loss_count)
            print('| epoch %d | perplexity %.2f'
                % (epoch+1, ppl))
            ppl_list.append(float(ppl))
            total_loss, loss_count = 0, 0

    #パラメータの保存
    with open('seq2seq.pkl', 'wb') as f:
        pickle.dump(model.params, f)

    # グラフの描画
    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label='train')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()

else:
    with open('seq2seq.pkl', 'rb') as f:
        model.param = pickle.load(f)

# #テストデータで評価
test_data = read_data(9,9, Flag='eval')
# test_inputlayer = InputLayer(test_data, 16, True)
#
test_data = inputlayer.test_dict_cre(test_data)

x = model.generate(np.array([test_data]), 1, 100)

text = ''
for id in x:
    text += inputlayer.all_id_to_word[id]

print(text)

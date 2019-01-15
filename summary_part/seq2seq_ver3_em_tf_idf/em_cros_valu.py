# coding: utf-8
import sys
sys.path.append('..')
from eval.similarity import *
from eval.cros import *
from eval.plot import *
from common import config
# GPUで実行する場合は下記のコメントアウトを消去（要cupy）
# ==============================================
# config.GPU = True
# ==============================================
from common.optimizer import Adam
from seq2seq_ver3_em_tf_idf.em_tf_idf_trainer import Trainer
from seq2seq_ver3_em_tf_idf.inputlayer import EM_TF_IDF_InputLayer
from seq2seq_ver3_em_tf_idf.emseq2seq import Seq2seq
from common.util import eval_perplexity, to_gpu

import numpy as np
import matplotlib.pyplot as plt
import pickle
from text_form import *
from text_form.eval import eval



def em_cros_valu(idx, wordvec_size, hidden_size, lr, max_epoch, create_fi, date, seed, learn=True):

    # ハイパーパラメータの設定
    batch_size = 3
    # wordvec_size = 50
    # hidden_size = 500
    time_size = 5  # Truncated BPTTの展開する時間サイズ
    # lr = 0.05
    # max_epoch = 300
    max_grad = 5.0
    #similarityのカウンタ
    simicount = 0

    #ファイルへのパラメータ書き込み
    fn = open('../eval/em/'+date+'.txt', "a")
    para = 'ハイパーパラメータの値\n\
batch_size = '+str(batch_size)+'\n\
wordvec_size = '+str(wordvec_size)+'\n\
hidden_size = '+str(hidden_size)+'\n\
time_size = '+str(time_size)+'\n\
lr = '+str(lr)+'\n\
max_epoch = '+str(max_epoch)+'\n\
max_grad = '+str(max_grad)+'\n'

    fn.write(para)
    fn.close()

    train, test, em, summary = eva_test_train(idx)

    #クロスバリデーション実行回数
    number = 1

    for (x, t, e, s) in zip(train, test, em, summary):
        # 学習時に使用する変数
        total_loss = 0
        loss_count = 0
        ppl_list = []

        inputlayer = EM_TF_IDF_InputLayer(t, 16, False)
        summary_corpora, _, _ = inputlayer.get_corpus()
        docu_ts = inputlayer.get_train_data(summary_corpora)

        inputlayer = EM_TF_IDF_InputLayer(x, 16, False)
        corpora, id_to_word, word_to_id = inputlayer.get_corpus()
        vocab_size = len(word_to_id)
        docu_xs = inputlayer.get_train_data(corpora)

        tf_idf = inputlayer.tf_idf()

        train_em = inputlayer.em(e)
        model = Seq2seq(vocab_size, wordvec_size, hidden_size, seed)
        optimizer = Adam()
        trainer = Trainer(model, optimizer, tf_idf, train_em)

        docu_xs = np.array(docu_xs)
        docu_ts = np.array(docu_ts)
        print(str(number)+'回目のクロスバリデーション!')
        if learn:
            acc_list = []
            for epoch in range(max_epoch):
                trainer.fit(docu_xs, docu_ts, word_to_id, max_epoch=1,batch_size=batch_size, max_grad=max_grad)

            if number == 1:
                create_fi.cros_loss(trainer.loss_list, first=True)
            else:
                create_fi.cros_loss(trainer.loss_list)

            if number == len(test):
                create_fi.loss_plot(len(test), 'EM-TF-IDF', wordvec_size)


            with open('seq2seq_ver'+ str(number) +'.pkl', 'wb') as f:
                pickle.dump(model.params, f)

        else:
            with open('seq2seq_ver'+ str(number) +'.pkl', 'rb') as f:
                model.param = pickle.load(f)

        generated_text = eval(inputlayer,model)

        simi = similarity(generated_text, s[0], word_to_id)
        simicount += simi
        text = '第'+str(number)+'回\n生成文：'+generated_text+'\n類似度：'+str(simi)+'\n'
        fn = open('../eval/em/'+date+'.txt', "a")
        fn.write(text)
        fn.close()

        number = number + 1

    #類似度の平均を取りリストに追加
    simi = simicount / len(test)
    create_fi.similist(simi, 2)

    return None

from em_cros_valu import *
from datetime import datetime

<<<<<<< HEAD
def cros_valu(idx, learn=True):
    
    # ハイパーパラメータの設定
    batch_size = 3
    wordvec_size = 100
    hidden_size = 50
    time_size = 5  # Truncated BPTTの展開する時間サイズ
    lr = 0.1
    max_epoch = 3
    max_grad = 5.0
    
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
        model = Seq2seq(vocab_size, wordvec_size, hidden_size)
        optimizer = Adam()
        trainer = Trainer(model, optimizer, tf_idf, train_em)
        
        docu_xs = np.array(docu_xs)
        docu_ts = np.array(docu_ts)
        print(docu_xs)
        print(str(number)+'回目のクロスバリデーション!')
        if learn:
            acc_list = []
            for epoch in range(max_epoch):
                trainer.fit(docu_xs, docu_ts, word_to_id, max_epoch=1,batch_size=batch_size, max_grad=max_grad)

                trainer.plot('seq2seq'+ str(number))
=======


idx = [1,3,5,7]
wvec = 5
hidden = 5
lr = 0.05
max_e = 25
x=None
>>>>>>> ea705547afb18b193e9332f8cefc9995adc45d58

for i in range(6):

<<<<<<< HEAD
        else:
            with open('seq2seq_ver'+ str(number) +'.pkl', 'rb') as f:
                model.param = pickle.load(f)
            
        generated_text = eval(inputlayer,model)
        
        simi = similarity(generated_text, s[0], word_to_id)
        text = '第'+str(number)+'回\n生成文：'+generated_text+'\n類似度：'+str(simi)+'\n'
        fn = open('../eval/log3.txt', "a")
        fn.write(text)    
        fn.close()
        
        number = number + 1
        
    trainer.plot('seq2seq'+ str(number))
        
    return None
            
idx = [1,3,5,7]
cros_valu(idx, learn=True)
    
=======
    date = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
    # if i == 0:
    #     cros_valu(idx, wvec, hidden, lr, max_e):
>>>>>>> ea705547afb18b193e9332f8cefc9995adc45d58

    em_cros_valu(idx, wvec, hidden, lr, max_e, x, date)

    wvec += 5
    hidden += 5

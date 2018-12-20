def cre_docu(number, train=True):
    document = []
    emdata = []
    summary = []
    for i in range(1,9):            
        if train:
            f1 = open('../../corpora/hori_F&Q/train/hori_corpus'+ str(i) + '.txt', "r")
        else:
            f1 = open('../../corpora/hori_F&Q/summary/hori_summary'+ str(i) + '.txt', "r")
        text= f1.read()
        
        if not i == number:
            document.append(text)
            emdata.append([text])
        else:
            summary.append(text)
        f1.close()
    return document ,emdata, summary

def eva_test_train(idx):
    train, test, em, summary = [], [], [], []
    for i in idx:
        #トレインのデータ
        tr, e, _ = cre_docu(i)
        #テスト、評価データ
        te, _, su = cre_docu(i, False)
        train.append(tr)
        test.append(te)
        em.append(e)
        summary.append(su)
        
    return train, test, em, summary
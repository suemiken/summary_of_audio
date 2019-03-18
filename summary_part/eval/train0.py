import sys
sys.path.append('..')
from eval.plot import Create_fi
from seq2seq_ver1.cros_valu import cros_valu
from seq2seq_ver2_tf_idf.tf_cros_valu import tf_cros_valu
from seq2seq_ver3_em_tf_idf.em_cros_valu import em_cros_valu
from datetime import datetime
import numpy as np

idx = [1, 2, 4, 3]
wvec = 40
hidden = 40
lr=0.08
# lr=0.20
# lr = 0.40
max_e = 300
# max_e = 250
# max_e = 200

paramstr = 'train_emVer2_lr_0.08_maxe_300_'

cre_fig = Create_fi()

for i in range(12):
    date = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
    # if i == 0:
    #     cros_valu(idx, wvec, hidden, lr, max_e):
    print('5-Gramのクロスバリデーション')
    seed = np.random.randint(500)
    cros_valu(idx, wvec, hidden, lr, max_e, cre_fig, date, seed)
    print('TF-IDFのクロスバリデーション')
    tf_cros_valu(idx, wvec, hidden, lr, max_e, cre_fig, date, seed)
    print('EM-TF-IDFのクロスバリデーション')
    em_cros_valu(idx, wvec, hidden, lr, max_e, cre_fig, date, seed)

    cre_fig.xlist(wvec)
    cre_fig.save('cros/loss/'+paramstr+date)

    wvec += 10
    hidden += 10

cre_fig.simi_plot()
cre_fig.save('cros/similarity/'+paramstr+date)

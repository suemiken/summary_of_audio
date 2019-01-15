from tf_cros_valu import *
from datetime import datetime



idx = [1,3,5,7]
wvec = 50
hidden = 50
lr = 0.1
max_e = 150

for i in range(6):
    date = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
    # if i == 0:
    #     cros_valu(idx, wvec, hidden, lr, max_e):
    cre_fig = Create_fi()

    cros_valu(idx, wvec, hidden, lr, max_e, cre_fig, date)
    tf_cros_valu(idx, wvec, hidden, lr, max_e, cre_fig, date)
    em_cros_valu(idx, wvec, hidden, lr, max_e, cre_fig, date)

    cre_fig.save(date)

    wvec += 25
    hidden += 25

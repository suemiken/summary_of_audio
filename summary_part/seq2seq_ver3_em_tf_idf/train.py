from em_cros_valu import *
from datetime import datetime



idx = [1,3,5,7]
wvec = 5
hidden = 5
lr = 0.05
max_e = 25
x=None

for i in range(6):

    date = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
    # if i == 0:
    #     cros_valu(idx, wvec, hidden, lr, max_e):

    em_cros_valu(idx, wvec, hidden, lr, max_e, x, date)

    wvec += 5
    hidden += 5

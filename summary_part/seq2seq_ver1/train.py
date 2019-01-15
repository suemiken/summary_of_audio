from cros_valu import *



idx = [1,3,5,7]
wvec = 50
hidden = 50
lr = 0.1
max_e = 200

for i in range(6):

    # if i == 0:
    #     cros_valu(idx, wvec, hidden, lr, max_e):

    cros_valu(idx, wvec, hidden, lr, max_e)

    wvec += 25
    hidden += 25

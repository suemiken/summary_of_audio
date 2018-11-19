import numpy as np
from dict_create import dict_cre
from format_text import format_text

test_data = open("/home/suemiken/デスクトップ/研究/教師データ/超一流の思考回路/落合陽一「これを理解してない人に危機感を感じます。」ネットの普及が原因！？ 2017.9.20 放送分/t_text", "r")

text = format_text(test_data)
f_dict, b_dict = dict_cre(text)

print(f_dict)

a = np.zeros(len(f_dict))
print(a)

a[b_dict['ストロー']] = 1
print(a)

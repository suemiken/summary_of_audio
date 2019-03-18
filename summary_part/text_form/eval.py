import sys
sys.path.append('..')
from text_form.read_text import read_data
import numpy as np

def eval(inputlayer, model):
    # #テストデータで評価
    test_data = read_data(9,9, Flag='eval')
    # test_inputlayer = InputLayer(test_data, 16, True)
    
    test_data = inputlayer.test_dict_cre(test_data)

    x = model.generate(np.array([test_data]), 1, 100)

    text = ''
    for id in x:
        text += inputlayer.all_id_to_word[id]
        
    return text

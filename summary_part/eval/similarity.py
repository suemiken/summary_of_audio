import numpy as np
from natto import MeCab

def similarity(text1, text2, word_to_id):
    
    size = len(word_to_id)
    textvec1 = np.zeros(size)
    textvec2 = np.zeros(size)
    
    textvec1 = wakati(text1, textvec1, word_to_id)
    textvec2 = wakati(text2, textvec2, word_to_id)
    
    norm1 = np.linalg.norm(textvec1)    
    norm2 = np.linalg.norm(textvec2)
    
    cos_simi = (np.dot(textvec1, textvec2))/ (norm1 * norm2)
    
    return cos_simi
    
def wakati(text, textvec, word_to_id):
    with MeCab('-F%m,%f[0],%h') as nm:
        for n in nm.parse(text, as_nodes=True):
            node = n.feature.split(',')
            if len(node) != 3:
                continue
            # print(node[0])
            if node[0] in word_to_id:
                textvec[word_to_id[node[0]]] = 1
    return textvec




# similarity('辛いけど頑張るホリエモン','私はホリエモン',{'私': 0, 'は': 1, 'ホリエモン' : 2, '辛い': 3, 'けど': 4, '頑張る': 5})
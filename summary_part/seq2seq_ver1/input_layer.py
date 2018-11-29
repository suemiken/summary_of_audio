# coding: utf-8
import sys
sys.path.append('..')
from common.base_model import BaseModel
import numpy as np
from common.time_layers import *
# from dict_create import vocab_dict_cre
from natto import MeCab
from text_form.format_text import format_text

#レイヤとしての役割だけでなく、入力データの整形も一役かっている。
class InputLayer:
    def __init__(self, documents, wordvec_size, test_flag):
        # 文書の解析
        self.documents = documents
        self.f_dict, self.all_word_to_id, self.all_id_to_word = {}, {}, {}
        self.corpus = []
        self.cache = 0

        #コーパス作成
        self.format_document()
        self.corpus = self.get_train_data(self.corpus)
        if not test_flag:
            #レイヤーの設定
            rn = np.random.randn
            vocab_size = len(self.all_word_to_id)
            embed_W = (rn(vocab_size, wordvec_size - 2) / 100).astype('f')
            self.embed = TimeEmbedding(embed_W)
            self.params = self.embed.params
            self.grads = self.embed.grads
            self.hs = None

    def get_corpus(self):
        return self.corpus, self.all_id_to_word, self.all_word_to_id

    def get_input_size(self, input_data):
        max_size = 0
        for data in input_data:
            if max_size < len(data):
                max_size = len(data)
        return max_size


    def get_train_data(self, input_data):
        outdata = []
        max_size = self.get_input_size(input_data)
        for one_corpus in self.corpus:
            array = one_corpus
            for i in range(max_size - len(one_corpus)):
                array.append(self.all_word_to_id['null'])
            outdata.append(array)
        return outdata


    def vocab_dict_cre(self, text):
        corpus = []
        with MeCab('-F%m,%f[0],%h') as nm:
            for n in nm.parse(text, as_nodes=True):
                node = n.feature.split(',')
                if len(node) != 3:
                    continue
                if not node[0] in self.f_dict:
                    self.all_word_to_id.update({node[0]:self.cache})
                    self.all_id_to_word.update({self.cache:node[0]})
                    self.cache += 1
                    self.f_dict.update({node[0]:1})
                else:
                    self.f_dict[node[0]] += 1
                corpus.append(self.all_word_to_id[node[0]])

        self.corpus.append(corpus)
        self.all_word_to_id.update({'unk':self.cache})
        self.all_id_to_word.update({self.cache:'unk'})

        self.all_word_to_id.update({'null':self.cache+1})
        self.all_id_to_word.update({self.cache+1:'null'})
        self.f_dict.update({'null':len(self.corpus)})

        return None

    def test_dict_cre(self, document):
        text = document[0]
        test_corpus = []
        with MeCab('-F%m,%f[0],%h') as nm:
            for n in nm.parse(text, as_nodes=True):
                node = n.feature.split(',')
                if len(node) != 3:
                    continue
                if not node[0] in self.all_word_to_id:
                    test_corpus.append(self.all_word_to_id['unk'])
                else:
                    test_corpus.append(self.all_word_to_id[node[0]])
        return test_corpus

    def format_document(self):
        long_text = ''
        art_text = ''
        copora = []
        for text in self.documents:
            art_text = format_text(text)
            self.vocab_dict_cre(art_text)
        return None

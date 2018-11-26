# coding: utf-8
import sys
sys.path.append('..')
from common.base_model import BaseModel
import numpy as np
# from dict_create import vocab_dict_cre
from natto import MeCab
from text_form.format_text import format_text


#レイヤとしての役割だけでなく、入力データの整形も一役かっている。
class InputLayer:
    def __init__(self, documents, wordvec_size, test_flag):
        # 文書の解析
        self.documents = documents
        self.f_dict, self.all_word_to_id, self.all_id_to_word = {}, {}, {}
        self.ETD_W = None
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
            self.TD_W = self.tf_idf()
            self.embed = TimeEmbedding(embed_W, self.TD_W)
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

    # [[[speaker_id, content_text, emotion],[speaker_id, content_text, emotion]],[[speaker_id, content_text, emotion],[speaker_id, content_text, emotion]]]
    def create_em_speaker_text(self):
        uttrance_em = []
        em_speakers_info = []
        text_number = 0
        for text in self.documents:
            text = format_text(text, replace_f=False)
            text = text.replace('話者:', '')
            text = text.replace(': ', '')
            text = text.replace('怒り', '0')
            text = text.replace('喜び', '1')
            text = text.replace('悲しみ', '2')
            text = text.replace('驚き', '3')
            text = text.replace('平静', '4')
            text = text.replace('恐れ', '5')
            uttrances = []
            uttrances = text.split('\n', -1)
            for uttrance in uttrances:
                uttrance_em.append(uttrance.split(' ', 2))
            em_speakers_info.append(uttrance_em)

        return em_speakers_info


    def tf_idf(self):
        tf_idf = []
        #領域の静的確保
        tf = [[0 for i in range(len(self.all_word_to_id))] for j in range(len(self.corpus))]
        idf = [0 for i in range(len(self.all_word_to_id))]
        #これでいいんか？？？？？？
        idf[-2] = 0.00001
        for i, one_corpus in enumerate(self.corpus):
            for id in one_corpus:
                tf[i][id] += 1
        ntf = np.array(tf)

        for art_tf in tf:
            for i, text_tf in enumerate(art_tf):
                if text_tf != 0:
                    idf[i] += 1
        for i, D in enumerate(idf):
            idf[i] = round(np.log(len(self.corpus) / D) + 1, 3)
        nidf = np.array(idf)

        for one_tf in ntf:
            tf_idf.append(nidf * one_tf)

        return tf_idf


    def em(self):
        #前者/後者
        #       怒り0  喜び1  悲しみ2  驚き3 平静4  恐れ5
        #怒り0   w11  w12   w13    w14  w15  w16
        #喜び1    w21  w22   w23    w24  w25  w26
        #悲しみ2  ・・・・・
        #驚き3    ・
        #平静4    ・
        #恐れ5    ・

        em_W = np.array([
            [ 0.1, 0.1, 0.1, 0.2, 0.3, 0.2],
            [ 0.4, 0.6, 0.5, 0.7, 0.6, 0.4],
            [ 0.3, 0.1, 0.5, 0.5, 0.5, 0.3],
            [ 0.4, 0.5, 0.2, 0.4, 0.8, 0.3],
            [ 0.7, 0.6, 0.5, 0.8, 0.6, 0.4],
            [ 0.1, 0.5, 0.4, 0.3, 0.5, 0.2]])

        em_speakers_docu = self.create_em_speaker_text()
        speaker_em_W = []
        speakers_em_W = []
        former_em = None
        for speakers_info in em_speakers_docu:
            for j, speaker_info in enumerate(speakers_info):
                if j == len(speakers_info):
                #youtube発話の最後は喜びであることが多いことを利用。
                    speaker_em_W.append(em_W[former_em][1])
                elif j == 0:
                    former_em = int(speaker_info[2])
                else:
                    after_em = int(speaker_info[2])
                    speaker_em_W.append(em_W[former_em][after_em])
                    former_em = after_em
            speakers_em_W.append(speaker_em_W)
        return speakers_em_W


    def update_docuement(self, text):
        self.documents.append(text)

        return None

    def tfidf_summary(self, num_sent):
        summary_sentence = []
        # self.documents.append(text)
        tf_idf = self.tf_idf()
        tar_tf_idf = tf_idf[-1]
        target_art = self.create_em_speaker_text()
        target_art = target_art[-1]
        score = [0 for i in range(len(target_art))]

        for id, sentence in enumerate(target_art):
            text = sentence[1]
            with MeCab('-F%m,%f[0],%h') as nm:
                for n in nm.parse(text, as_nodes=True):
                    node = n.feature.split(',')
                    if len(node) != 3:
                        continue
                    if self.all_word_to_id.get(node[0]) == None:
                        continue
                    score[id] += tar_tf_idf[self.all_word_to_id.get(node[0])]

        score = np.array(score)
        max_sort = np.argsort(score)[::-1]
        for i, index in enumerate(max_sort):
            if num_sent == i:
                break
            summary_sentence.append(target_art[index][1])
        return summary_sentence

# -*- coding: utf-8 -*-
import numpy as np
from dict_create import vocab_dict_cre
from format_text import format_text

documents = []
text1 = open("../practice_corpora/corpora/hori_F&Q/hori_corpus01.txt", "r")
text2 = open('../practice_corpora/corpora/hori_F&Q/hori_corpus02.txt', "r")


documents.append(text1)
documents.append(text2)


class EM_TF_IDF:
    def __init__(self, documents):
        self.documents = documents
        self.f_dict, self.all_word_to_id, self.all_id_to_word = None, None, None

    def format_document(self):
        long_text = ''
        for text in self.documents:
            long_text += format_text(text)

        self.f_dict, self.all_word_to_id, self.all_id_to_word = vocab_dict_cre(long_text)

        return None

    # [[[speaker_id, content_text, emotion],[speaker_id, content_text, emotion]],[[speaker_id, content_text, emotion],[speaker_id, content_text, emotion]]]
    def create_em_speaker_text(self):
        uttrance_em = []
        em_speakers_info = []
        text_number = 0
        for text in self.documents:
            text = format_text(text,replace_f=False)
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

    def tf(self):
        tf = np.zeros(len(self.all_word_to_id))
        for word, fre in self.f_dict.items():
                #たんごidを頻度に変換するdict
                tf[self.all_word_to_id[word]] += fre

        return tf

    def idf(self):
        idf = np.zeros(len(self.all_word_to_id))
        for id, word in self.all_id_to_word.items():
            for text in self.documents:
                f, word_to_id, id_to_word = vocab_dict_cre(text)
                if word in word_to_id:
                    idf[id] += 1

        return idf

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
            [ 0.1, 0.1, 0.1,  0.2,  0.3, 0.2],
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

weight = EM_TF_IDF(documents)
# weight.format_document()
# print(weight.tf())
#print(weight.create_em_speaker_text())
print(weight.em())

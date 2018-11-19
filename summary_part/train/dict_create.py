# -*- coding: utf-8 -*-
#!/usr/bin/env python

from natto import MeCab
from format_text import format_text


def vocab_dict_cre(text):
    i = 0
    frequent_dict = {}
    word_to_id = {}
    id_to_word = {}
    corpus = []
    with MeCab('-F%m,%f[0],%h') as nm:
        for n in nm.parse(text, as_nodes=True):
            node = n.feature.split(',')
            if len(node) != 3:
                continue
            if not node[0] in frequent_dict:
                word_to_id.update({node[0]:i})
                id_to_word.update({i:node[0]})
                i += 1
                frequent_dict.update({node[0]:1})
            else:
                frequent_dict[node[0]] += 1
            corpus.append(word_to_id[node[0]])
        return corpus, frequent_dict, word_to_id, id_to_word




# -F / --node-format オプションでノードの出力フォーマットを指定する
#
# %m    ... 形態素の表層文
# %f[0] ... 品詞
# %h    ... 品詞 ID (IPADIC)
# %f[8] ... 発音
#
# #単語を登録する辞書型
# def 1-of-N(text, dict)

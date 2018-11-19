# coding: utf-8
import sys
sys.path.append('..')

def get_data():
    documents = []
    for i in range(1,9):
        f = open('../../corpora/hori_F&Q/hori_corpus0'+ str(i) + '.txt', "r")
        text= f.read()
        f.close()
        documents.append(text)

    return documents

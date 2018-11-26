# coding: utf-8
import sys
sys.path.append('..')

def read_data(first, last, Flag='train'):
    summary_documents = []
    if Flag == 'train':
        for i in range(first,last):
            f = open('../../corpora/hori_F&Q/train/hori_corpus'+ str(i) + '.txt', "r")
            text= f.read()
            f.close()
            summary_documents.append(text)
    elif Flag == 'test':
        for i in range(first,last):
            f = open('../../corpora/hori_F&Q/summary/hori_summary'+ str(i) + '.txt', "r")
            text= f.read()
            f.close()
            summary_documents.append(text)
    elif Flag == 'eval':
        f = open('../../corpora/hori_F&Q/train/hori_corpus'+ str(last) + '.txt', "r")
        text= f.read()
        f.close()
        text = format_text(text)
        summary_documents.append(text)
    return summary_documents

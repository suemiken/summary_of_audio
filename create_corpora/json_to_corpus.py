import sys
import json
import re
import glob

def json_to_corpus():

    #最終的にはディレクトリかにあるjsonファイルを読み込みコーパスを作成。
    in_file_list = glob.glob('json_from_YouTube/hori_F&Q/*.json')


    dicts_from_json = []
    for in_file in in_file_list:
        in_file = open(in_file, 'r')
        dict_from_json = json.load(in_file)
        dicts_from_json.append(dict_from_json)
        in_file.close()

    final_t = True
    final_f = False

    for i, dict in enumerate(dicts_from_json):
        timestamps = []
        speaker_labels = []
        speaker_labels = dict['speaker_labels']
        for j in range(len(dict['results'])):
            for k in range(len(dict['results'][j]['alternatives'])):
                m = 1
                for word_info in (dict['results'][j]['alternatives'][k]['timestamps']):
                    if m == len(dict['results'][j]['alternatives'][k]['timestamps']):
                        final_word = word_info
                        timestamps.append(final_word+[final_t])
                    else:
                        word = word_info
                        timestamps.append(word+[final_f])
                        m += 1

                    #speakerのid配列を作成
        speaker_id = []
        for speaker in speaker_labels:
            speaker_id += str(speaker['speaker'])

#コーパスに入力するテキストの作成
        corpus_text = ''
        first = True
        for j, timestamp in enumerate(timestamps):
            if first:
                corpus_text += '話者:' + speaker_id[j] + ' '
                first = False
                change_f = True
            if timestamp[3] == final_t:
                corpus_text += timestamp[0] + ' : \n'
                first = True
                corpus_text = corpus_text.replace('D_',' : \n話者:' + speaker_id[j] + ' ')

            else:
                corpus_text += timestamp[0]
            if change_f:
                corpus_text = corpus_text.replace('D_','')
                change_f = False
            else:
                corpus_text = corpus_text.replace('D_',' : \n話者:' + speaker_id[j] + ' ')

        corpus_text += corpus_text.rstrip('\n')
        out_file = open('corpora/hori_F&Q/hori_corpus' +str(i+1) + '.txt', 'w', encoding='utf-8')
        out_file.write(corpus_text)
        out_file.close()

    return None

json_to_corpus()

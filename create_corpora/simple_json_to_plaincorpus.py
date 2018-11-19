import sys
import json
import re
import glob


 #最終的にはディレクトリかにあるjsonファイルを読み込みコーパスを作成。
in_file_list = glob.glob('json_from_YouTube/*.json')

dicts_from_json = []
for in_file in in_file_list:
    in_file = open(in_file, 'r')
    dict = json.load(in_file)
    dicts_from_json.append(dict)
    in_file.close()

#print(dict['results'][0]['alternatives'][0]['transcript'])
sentence = ''
array = []
befere=' '
after=''

for dict in dicts_from_json:
    for j in range(len(dict['results'])):
        for k in range(len(dict['results'][j]['alternatives'])):
            sentence += dict['results'][j]['alternatives'][k]['transcript'] + '\n'

        sentence = re.sub(befere, after, sentence)
    sentence = sentence.rstrip('\n')
    array.append(sentence)
    sentence = ''



number=1

for sentence in array:
    str_number = str(number)
    out_file = open('corpora/plain_corpus' + str_number + '.txt', 'w', encoding='utf-8')
    out_file.write(sentence)
    out_file.close()
    number += 1

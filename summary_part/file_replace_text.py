import re
import glob

def replace_text(path, befere, after):
    f = open(path, 'r')
    lines = f.read()
    text = re.sub(befere, after, lines)
    f.close()
    f = open(path, 'w')
    f.write(text)
    f.close()

    return

#file_replace    
in_file_list = glob.glob('../practice_corpora/learing_data/wkidata/1~49/taihi/*.txt')


error_words = ['①', '②', '③', '④', '⑨', 'キェ', 'ソソル・キェボ', 'トーーク', 'ー', 'ーア・ブッセ', 'ーイ', 'ーウ', 'ーオル', 'ーグ', 'ーコフ', 'ーサ', 'ース', 'ーズ', 'ーズ・ドートリッシュ',
    'ーセック', 'ーチウム', 'ーデ', 'ーデント', 'ート', 'ード', 'ーナ', 'ーナー', 'ーヌ', 'ーネ',' ーノ','ープス', 'ーペ', 'ーム', 'ーユ', 'ーラ', 'ーリ', 'ーリィ', 'ール', 'ールバニ', 'ーン',
    'ーヴ', 'ーヴィチ', 'ー・ゼル', 'ーーーー']

for i, in_file in enumerate(in_file_list):
    in_file = open(in_file, 'r')
    lines = in_file.read()
    path = '../practice_corpora/learing_data/wkidata/1~49/taihi/wki'+str(i)+'.txt'
    for error_word in error_words:
        text = re.sub(error_word, '', lines)
    in_file.close()
    f = open(path, 'w')
    f.write(text)
    f.close()

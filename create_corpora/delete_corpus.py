from watson_developer_cloud import SpeechToTextV1
import json

speech_to_text = SpeechToTextV1(
    username='be060cc5-6111-4a6b-9231-d223c218c082',
    password='EqlZhVwCoWr7')


customization_id = "0f084cc6-3382-4fbf-a36c-8774d19f073c"


def mul_de_courpus(id, n = 1):
    for i in range(4):
        num = n + i
        speech_to_text.delete_corpus(
            id,
            'corpus_'+str(num)
        )


error_words = ['①', '②', '③', 'キェ', 'ソソル・キェボ', 'トーーク', 'ー', 'ーア・ブッセ'
,'ーイ', 'ーウ', 'ーオル', 'ーガ', 'ークタム', 'ーグ', 'ージュ', 'ース', 'ーズ', 'ーズ・ドートリッシュ', 'ーセック', 'ーソフ', 'ーデ', 'ーデント', 'ート', 'ード', 'ーナ','ーナー', 'ーヌ', 'ーネ',
'ーノ', 'ーペ', 'ーム', 'ーモ', 'ーヤ', 'ーユ', 'ーラ', 'ーリ', 'ール', 'ールバニ', 'ーン', 'ーヴ', 'ーヴィチ', 'ー・ゼル', 'ー・ベ']

def word_replace(id, error_words):
    for error_word in error_words:
        speech_to_text.delete_word(
            id,
            error_word
        )
word_replace(customization_id, error_words)

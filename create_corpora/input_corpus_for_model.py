from watson_developer_cloud import SpeechToTextV1
import sys
import json
from os.path import dirname, join
import glob
import time


speech_to_text = SpeechToTextV1(
    username='be060cc5-6111-4a6b-9231-d223c218c082',
    password='EqlZhVwCoWr7')


acoustic_flag = True
# 'af52ea45-9bd3-4781-96aa-428ad81b1f31
if acoustic_flag:
    customization_id = '0a054c42-5ba7-4ed6-81de-6b1de19d3ee6'
    audio_file_list = glob.glob('audio/hori_F&Q/*.mp3')

    for i, audio_file in enumerate(audio_file_list):
        audio_file = open(audio_file, 'rb')
        speech_to_text.add_audio(
                customization_id,
                'audio'+str(i+19),
                audio_file,
                'audio/mp3')
        audio_file.close()
        time.sleep(40.0)
else:
    customization_id = "0f084cc6-3382-4fbf-a36c-8774d19f073c"
    file_list = glob.glob('./learing_data/wkidata/1~49/*.txt')
    for i, file in enumerate(file_list):
        corpus_file = open(file, 'rb')
        speech_to_text.add_corpus(
                customization_id,
                'corpus_'+str(i+5),
                corpus_file)
        corpus_file.close()
        time.sleep(200.0)

# -*- coding: utf-8 -*-

from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback, AudioSource
from os.path import join, dirname
import json
import re
import glob
import time



class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_data(self, data):
        print(json.dumps(data, indent=2))

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

# myRecognizeCallback = MyRecognizeCallback()

def data_collect():
    speech_to_text = SpeechToTextV1(
        username='be060cc5-6111-4a6b-9231-d223c218c082',
        password='EqlZhVwCoWr7')

    for i in range(8):
        with open(join(dirname(__file__), './../../audio/hori_FQ', 'hori'+ str(i+13) +'.mp3'), 'rb') as audio_file:
            speech_result = speech_to_text.recognize(
                    audio=audio_file,
                    content_type='audio/mp3',
                    timestamps=True,
                    customization_id='0f084cc6-3382-4fbf-a36c-8774d19f073c',
                    acoustic_customization_id='0a054c42-5ba7-4ed6-81de-6b1de19d3ee6',
                    model='ja-JP_BroadbandModel',
                    customization_weight=0.4,
 #                  recognize_callback=myRecognizeCallback,
 #                  word_alternatives_threshold=0.9,
                    speaker_labels=True).get_result()

        time.sleep(600.0)
        with open(join(dirname(__file__), './json_from_YouTube/hori_FQ', 'hori_FQ'+ str(i+13) + '.json'), 'w') as file:
            data = json.dumps(speech_result, indent=2, ensure_ascii=False)
            file.write(data)

data_collect()

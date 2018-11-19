# -*- coding: utf-8 -*-


from watson_developer_cloud import SpeechToTextV1
from watson_developer_cloud.websocket import RecognizeCallback, AudioSource
from os.path import join, dirname
import json

speech_to_text = SpeechToTextV1(
    username='be060cc5-6111-4a6b-9231-d223c218c082',
    password='EqlZhVwCoWr7')

# corpora = speech_to_text.list_corpora('ja-JP_NarrowbandModel').get_result()
# print(json.dumps(corpora, indent=2))

speech_model = speech_to_text.get_language_model('0f084cc6-3382-4fbf-a36c-8774d19f073c').get_result()
print(json.dumps(speech_model, indent=2))

# -*- coding: utf-8 -*-

from watson_developer_cloud import SpeechToTextV1
import json

speech_to_text = SpeechToTextV1(
    username='be060cc5-6111-4a6b-9231-d223c218c082',
    password='EqlZhVwCoWr7')

acoustic_flag = True

if acoustic_flag:
    # acoustic_models
    # "6c7a294f-c13d-43c6-bcee-960cedd5baa7"
    model = speech_to_text.create_acoustic_model(
        'hori acoustic model',
        'ja-JP_BroadbandModel',
        description='hori custom acoustic model'
    ).get_result()
else:
# "customization_id": "0f084cc6-3382-4fbf-a36c-8774d19f073c"
#失敗　"46b4ee56-6afc-4ad6-ba12-120eed2c64ee"
    model = speech_to_text.create_language_model(
        'Japanise language model',
        'ja-JP_BroadbandModel',
        description='Japanise cosutom language model').get_result()


print(json.dumps(model, indent=2))

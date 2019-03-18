
from watson_developer_cloud import SpeechToTextV1
import json

speech_to_text = SpeechToTextV1(
    username='be060cc5-6111-4a6b-9231-d223c218c082',
    password='EqlZhVwCoWr7')


customization_id = "0f084cc6-3382-4fbf-a36c-8774d19f073c"
# speech_to_text.train_language_model(customization_id)
# language_model = speech_to_text.get_language_model(customization_id).get_result()
# print(json.dumps(language_model, indent=2))


#acoustic
acoustic_customization_id = '0a054c42-5ba7-4ed6-81de-6b1de19d3ee6'
speech_to_text.train_acoustic_model(acoustic_customization_id)
# speech_to_text.upgrade_acoustic_model(acoustic_customization_id)

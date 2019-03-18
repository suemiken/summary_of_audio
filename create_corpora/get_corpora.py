from watson_developer_cloud import SpeechToTextV1
import json

speech_to_text = SpeechToTextV1(
    username='be060cc5-6111-4a6b-9231-d223c218c082',
    password='EqlZhVwCoWr7')

customization_id = "0f084cc6-3382-4fbf-a36c-8774d19f073c"



# corpora = speech_to_text.list_corpora(customization_id).get_result()
# print(json.dumps(corpora, indent=2))
# model = speech_to_text.get_language_model(customization_id).get_result()
# print(json.dumps(model, indent=2))


audio_customization_id = '0a054c42-5ba7-4ed6-81de-6b1de19d3ee6'

# speech_to_text.reset_acoustic_model(audio_customization_id)

# acoustic_model
audio_resources = speech_to_text.list_audio(audio_customization_id).get_result()
print(json.dumps(audio_resources, indent=2))
# #
acoustic_model = speech_to_text.get_acoustic_model(audio_customization_id).get_result()
print(json.dumps(acoustic_model, indent=2))

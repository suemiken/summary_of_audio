import re

def format_text(text, replace_f=True):
    if replace_f:
        text = text.replace('\n', '')
        text = re.sub('話者:0 ', '', text)
        text = re.sub('話者:1 ', '', text)
        text = re.sub('話者:2 ', '', text)
        text = re.sub('話者:3 ', '', text)

        text = re.sub(' : 平静', '', text)
        text = re.sub(' : 驚き', '', text)
        text = re.sub(' : 悲しみ', '', text)
        text = re.sub(' : 怒り', '', text)
        text = re.sub(' : 喜び', '', text)

    return text
#
# print(format_text(test_data))

#
# dict={}
# i = 0
#
# with MeCab('-F %m') as nm:
#     for line in lines:
#
#         text = line.replace('\n','')
#         text = re.sub('EOS','',text)
#         print(nm.parse(text))
# #         dict.update({nm.parse(text):i})
# #         i += 1
# #
# # print(dict)

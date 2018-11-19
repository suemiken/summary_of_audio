import numpy as np

in_file = open('learing_data/kwic1.txt', 'r')

lines = in_file.read()
aray = lines.split('	')

str = ''

i = 0
while(i+1 < len(aray)/23):
    str += aray[5+23*i]
    str += '#'
    str += aray[7+23*i]
    str += '#'
    i += 1

str = str.replace('#','\n')

out_file = open('corpora/for_ibm_corpus1.txt', 'w')
out_file.write(str)
out_file.close()

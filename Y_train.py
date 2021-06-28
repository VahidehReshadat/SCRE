import json
import warnings; warnings.filterwarnings(action='ignore', category=Warning)
from kargen.models import SequenceModel
from kargen.preprocessing import load_data_and_labels
import numpy as np
from keras.utils import to_categorical
import re
import pandas as pd
from pickle import dump
from keras.utils import np_utils
#from keras.layers import Dense
from keras.layers import TimeDistributed
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from string import printable


def process_sent(sent):
    if sent not in [" ", "\n", ""]:#yani agar sent khali nist
        sent = sent.strip("\n")
        sent = re.sub('<[A-Z]+/*>', '', sent) # remove special tokens eg. <FIL/>, <S>
        sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|♫:;—”“~`#]", " ", sent)
        sent = re.sub(' {2,}', ' ', sent) # remove extra spaces > 1
        sent = re.sub("^ +", "", sent) # remove space in front
        sent = re.sub(r"([\.\?,!]){2,}", r"\1", sent) # remove multiple puncs
        sent = re.sub(r" +([\.\?,!])", r"\1", sent) # remove extra spaces in front of punc
        #sent = re.sub(r"([A-Z]{2,})", lambda x: x.group(1).capitalize(), sent) # Replace all CAPS with capitalize

        #sent = re.sub(sent).strip()
        sent = re.sub("[^{}]+".format(printable), "", sent)
        sent = re.sub('\W+', '', sent).strip()
        #sent = sent.strip()
        #sent = sent.replace()
        #sent = ''.join(sent.split())





        return sent
    return


#data_path = './data/FinalData/NEW_MyTRAIN_FILE_Ver60.TXT'
data_path = './data/FinalData/Yall/Y_all_dataset1.TXT'

with open(data_path, 'r', encoding='utf8') as f:
    text = f.readlines()
mg = int(len(text) / 4)
# print(mg)

relations, comments, blanks = [], [], []
sents = list()

for i in range(int(len(text) / 4)):
    sent = text[4 * i]
    sent = re.findall("\"(.+)\"", sent)[0]  # Regx101 haechizi e beine "" ast
    # sent = re.findall("\"(.+)\"", sent)[0] #Regx101 haechizi e beine "" ast
    # print(sent)
    sent = sent.replace('<e1>', '')
    sent = sent.replace('</e1>', '')
    sent = sent.replace('<e2>', '')
    sent = sent.replace('</e2>', '')

    ################################## X Process
    sent=process_sent(sent)

    sents.append(sent)
    relation = text[4 * i + 1]
    #relation = relation.rstrip()

################################## Y Process
    relation=process_sent(relation)

    relations.append(relation)
    comment = text[4 * i + 2]
    blank = text[4 * i + 3]

d = {'sentences': sents, 'labels': relations}
df = pd.DataFrame(data=d)
print(df)

#dump(df['labels'].values, open('lis_Y_all_Train_Ds1_small.pkl', 'wb'))
#dump(df['labels'].values, open('lis_Y_all_Test_Ds1_small.pkl', 'wb'))
dump(df['labels'].values, open('YallDs1.pkl', 'wb'))



print("len(df['labels'].values)")
#print(len(df['labels'].values))

print(type(df['labels'].values))
print(type(df['labels'].values[0]))

#print(len(df['labels'].values[1]))


'''
encoder = LabelEncoder()
encoder.fit(df['labels'].values)
encoded_Y = encoder.transform(df['labels'].values)
# convert integers to dummy variables (i.e. one hot encoded)
print("num_classssssssssssssssssssssssssssss")
#num_class=encoder.classes_()
#print(num_class)
dummy_y = np_utils.to_categorical(encoded_Y)

'''


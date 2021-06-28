
import json
import warnings; warnings.filterwarnings(action='ignore', category=Warning)
from kargen.models import SequenceModel
from kargen.preprocessing import load_data_and_labels
import numpy as np
from keras.utils import to_categorical
import re
from pickle import dump
import pandas as pd


'''
x_train, y_ner_train, y_term_train, y_rel_train = load_data_and_labels(f"data/kargo/train/kpm_terms_only.txt")
x_dev, y_ner_dev, y_term_dev, y_rel_dev = load_data_and_labels("data/kargo/dev_rel.txt")
x_test, y_ner_test, y_term_test, y_rel_test = load_data_and_labels("data/kargo/test_rel.txt")
x_online, y_ner_online, y_term_online, y_rel_online = load_data_and_labels("data/kargo/online_rel.txt")

print(len(x_train))
print(len(y_ner_train))

'''

data_path = './data/MyData/NEW_MyTRAIN_FILE_Ver60.TXT'
#data_path = './data/MyData/NEW_TEST_FILE_FULL30.TXT'


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
    sents.append(sent)
    relation = text[4 * i + 1]
    relation = relation.rstrip()
    relations.append(relation)
    comment = text[4 * i + 2]
    blank = text[4 * i + 3]

# print(sents)

#X_my_Train = np.array(sents)
# print("X_my_Trainnnnn")
# print(X_my_Train)
#print("X_my_Train[0]")
#print(X_my_Train[0])

model_old = SequenceModel.load(
   weights_file="pretrain_models/lstm/dev_40e/weights.h5",
   preprocessor_file="pretrain_models/lstm/dev_40e/preprocessors.json"
)

#model_old=SequenceModel()

List_X_Train_3embed= list()


#for i, val in enumerate(sents):
#    print(i, val)

#for j in sents:
#    print(j)


print(len(sents))

for i in sents:
    print(i)
    analysis, preproc = model_old.analyze(i)
    print(analysis)
    print("type(analysis)")

    print(type(analysis))

    print("type(preprocsssssssssssssssssssssssssssssssssss)")
    #print(type(preproc))
    #print(len(preproc))
    print(preproc[0].shape)
    print(preproc[1].shape)
    print(preproc[2].shape)

    #myorder = [1, 0, 2]
    #mylist = [preproc[i] for i in myorder]

    #print(preproc.shape)
    #print("new listtttttttttttttttttt mylist")
    #print(mylist[0].shape)
    #print(mylist[1].shape)
    #print(mylist[2].shape)

    #List_X_Train_3embed.append(mylist)
    List_X_Train_3embed.append(preproc)



print("Akharrrr len(sents)")
print(len(List_X_Train_3embed))
print("List_X_Train_3embed[0][0].shape")

print(List_X_Train_3embed[0][0].shape)
#dump(List_X_Train_3embed, open('lis_X_all_Test.pkl', 'wb'))
dump(List_X_Train_3embed, open('lis_X_all_Train.pkl', 'wb'))


#object = pd.read_pickle(r'filepath')
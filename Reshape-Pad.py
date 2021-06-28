import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import Input, Model
from keras.layers import Embedding, TimeDistributed, Bidirectional, Flatten, LSTM, Concatenate, Dropout, Dense, Conv1D, \
    GlobalMaxPooling1D
from keras.models import Sequential
import numpy as np

############################## X_Train ############################

############################## X_Test ############################

X_Test_3embed = pd.read_pickle("data/FinalData/Dataset1(small)/X_Test_Ds1Small.pkl")  # ta sample len 1 ba shape (1,5,128), (1,16,128), etc...

print(type(X_Test_3embed))

max_len_second_dimention=max([i.shape[1] for i in X_Test_3embed])
print("max_len_second_dimention")
print(max_len_second_dimention)

Xt_Same_all_shape=[]
for k in X_Test_3embed:
    b = np.hstack([k, np.zeros([1, max_len_second_dimention - k.shape[1], 128])])
    Xt_Same_all_shape.append(b)

print(len(Xt_Same_all_shape))

print(Xt_Same_all_shape[0].shape[1])

for i in Xt_Same_all_shape:
    print(i.shape)

X_Train_3embed = np.array(Xt_Same_all_shape)

print("X-Train")
print(X_Train_3embed.shape)
print(type(X_Train_3embed))
print(X_Train_3embed[2].shape) # (1, 62, 128)

for i in X_Train_3embed:
    print(i.shape)
print(type(X_Train_3embed[0]))



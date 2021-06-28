import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import Input, Model
from keras.layers import Embedding, TimeDistributed, Bidirectional, LSTM, Concatenate, Dropout, Dense, Conv1D, \
    GlobalMaxPooling1D
from keras.models import Sequential
import numpy as np

############################## X_Train ############################

X_Train_3embed1 = pd.read_pickle("XX_Train_3embeding.pkl")  #230 ta sample len 1 ba shape (1,5,128)
#X_Train_3embed1 = pd.read_pickle("X_Train_3embeding.pkl")  #230 ta sample len 1 ba shape (1,5,128)


X_Train_3embed = np.array(X_Train_3embed1)

print("X-Train")
print(X_Train_3embed.shape)
print(type(X_Train_3embed))
print(X_Train_3embed[0].shape) # (1, 128)
print(type(X_Train_3embed[0]))


############################## X_Test ############################

X_Test_3embed1 = pd.read_pickle("XX_Test_3embeding.pkl")  #104 ta sample len 1 ba shape (1,32,128)
#X_Test_3embed1 = pd.read_pickle("X_Test_3embeding.pkl")  #104 ta sample len 1 ba shape (1,32,128)


X_Test_3embed=np.array(X_Test_3embed1)
print("X-Test")

print(X_Test_3embed.shape)
print(type(X_Test_3embed.shape))
print(X_Test_3embed[0].shape) #(1, 128)
print(type(X_Test_3embed[0]))


#print(type(X_Test_3embed)) #<class 'numpy.ndarray'>
#'''
############################## Y_Train ############################

Y_Train_labels_list = pd.read_pickle("lis_Y_all_Train.pkl")
encoder = LabelEncoder()
encoder.fit(Y_Train_labels_list)
encoded_Y = encoder.transform(Y_Train_labels_list)
Y_my_Train1 = np_utils.to_categorical(encoded_Y)

Y_my_Train = np.array(Y_my_Train1)

print("Y-Train")
print(Y_my_Train.shape)
print(type(Y_my_Train.shape))
print(Y_my_Train[0].shape)
print(type(Y_my_Train[0]))

############################## Y_Test ############################

X_Test_labels_list = pd.read_pickle("lis_Y_all_Test.pkl")
encoder = LabelEncoder()
encoder.fit(X_Test_labels_list)
encoded_Y = encoder.transform(X_Test_labels_list)
Y_my_Test1 = np_utils.to_categorical(encoded_Y)

Y_my_Test = np.array(Y_my_Test1)


print("Y-Test")
print(Y_my_Test.shape)
print(type(Y_my_Test))
print(Y_my_Test[0].shape)
print(type(Y_my_Test[0]))
######################  Model
#elmo_embeddings = Input(shape=(None, 1024), dtype='float32')

first_input = Input(shape=(230, ))

first_dense = Dense(128)(first_input)

output_layer = Dense(83, activation='softmax')(first_dense)

model = Model(inputs=first_input, outputs=output_layer)

model.summary()


'''
model = Sequential()
model.add(Input(shape=(128,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(83, activation='softmax'))
model.summary()
'''


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


history = model.fit((X_Train_3embed, Y_my_Train), epochs=2, batch_size=32)

tst_results = model.evaluate(X_Test_3embed, Y_my_Train, verbose=2)
print("Test LOSS:", tst_results[0])
print("Test Accuracy:", tst_results[1])

#'''
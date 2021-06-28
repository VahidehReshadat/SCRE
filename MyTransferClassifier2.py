import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import Input, Model
from keras.layers import Embedding, TimeDistributed, Bidirectional, Flatten, LSTM, Concatenate, Dropout, Dense, Conv1D, \
    GlobalMaxPooling1D
from keras.models import Sequential
import numpy as np

############################## X_Train ############################

X_Train_3embed = pd.read_pickle("XX_Train_3embeding.pkl")  #230 ta sample len 1 ba shape (1,5,128)
#X_Train_3embed = pd.read_pickle("X_Train_3embeding.pkl")  #230 ta sample len 1 ba shape (1,5,128)

#Convert list to numpy array
X_Train_3embed = np.array(X_Train_3embed)

print("X-Train")
print(X_Train_3embed.shape)
print(type(X_Train_3embed))
print(X_Train_3embed[2].shape) # (1, 128)

for i in X_Train_3embed:
    print(i.shape)
print(type(X_Train_3embed[0]))
#X_Train_3embed1 = list(X_Train_3embed)
#print(type(X_Train_3embed1))


############################## Y_Train ############################

Y_Train_labels_list = pd.read_pickle("lis_Y_all_Train.pkl")

print(type(Y_Train_labels_list))
print(type(Y_Train_labels_list[0]))

#Y_Train_labels_list = np.array(Y_Train_labels_list1)

encoder = LabelEncoder()
encoder.fit(Y_Train_labels_list)
encoded_Y = encoder.transform(Y_Train_labels_list)
Y_my_Train = np_utils.to_categorical(encoded_Y)


print("Y-Train")
print(Y_my_Train.shape)
print(type(Y_my_Train))
print(Y_my_Train[0].shape)
print(type(Y_my_Train[0]))

##################################  Model

#elmo_embeddings = Input(shape=(None, 1024), dtype='float32')

first_input = Input(shape=(1, 128))
first_dense = Dense(128)(first_input)
output_layer = Dense(83, activation='softmax')(first_dense)
outputs = Flatten()(output_layer)

model = Model(inputs=first_input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_Train_3embed, Y_my_Train, epochs=2, batch_size=32)


first_input = Input(shape=(1, 128))
first_dense = Dense(128)(first_input)
output_layer = Dense(83, activation='softmax')(first_dense)
outputs = Flatten()(output_layer)

model = Model(inputs=first_input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_Train_3embed, Y_my_Train, epochs=2, batch_size=32)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, recall_score
#from seqeval.metrics import precision_score, recall_score, f1_score

from keras.utils import np_utils
from keras import Input, Model
from keras.layers import Embedding, TimeDistributed, Bidirectional, Flatten, LSTM, Concatenate, Dropout, Dense, Conv1D, \
    GlobalMaxPooling1D
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
import numpy as np
import warnings;

warnings.filterwarnings(action='ignore', category=Warning)

############################## X_Train ############################

X_Train_3embed = pd.read_pickle("data/FinalData/Dataset1(small)/X_Train_3embeding.pkl")  #230 ta sample len 1 ba shape (1,5,128), (1,16,128), etc...

print(type(X_Train_3embed))

max_len_second_dimention=max([i.shape[1] for i in X_Train_3embed])
print("max_len_second_dimention")
print(max_len_second_dimention)

X_Same_all_shape=[]
for k in X_Train_3embed:
    b = np.hstack([k, np.zeros([1, max_len_second_dimention - k.shape[1], 128])])
    X_Same_all_shape.append(b)

print(len(X_Same_all_shape))

print(X_Same_all_shape[0].shape[1])

for i in X_Same_all_shape:
    print(i.shape)

X_Train_3embed = np.array(X_Same_all_shape)

print("X-Train")
print(X_Train_3embed.shape)
print(type(X_Train_3embed))
print(X_Train_3embed[2].shape) # (1, 62, 128)

for i in X_Train_3embed:
    print(i.shape)
print(type(X_Train_3embed[0]))



XX_Train_3embeding=[]

for i in X_Train_3embed:
    result = i[:, 0, :]
    print(result.shape)
    #c=i.astype(object)
    #result=np.squeeze(result)
    #print(result.shape)
    XX_Train_3embeding.append(result)


XX_Train_3embeding = np.array(XX_Train_3embeding)


############################## X_Test ############################

Xt_Train_3embed = pd.read_pickle("data/FinalData/Dataset1(small)/X_Test_3embeding.pkl")  #230 ta sample len 1 ba shape (1,5,128), (1,16,128), etc...

print(type(Xt_Train_3embed))

Xt_Same_all_shape=[]
for k in Xt_Train_3embed:
    b = np.hstack([k, np.zeros([1, 62 - k.shape[1], 128])])
    Xt_Same_all_shape.append(b)

print(len(Xt_Same_all_shape))

print(Xt_Same_all_shape[0].shape[1])

for i in Xt_Same_all_shape:
    print(i.shape)

Xt_Train_3embed = np.array(Xt_Same_all_shape)

print("X-Test")
print(Xt_Train_3embed.shape)
print(type(Xt_Train_3embed))
print(Xt_Train_3embed[2].shape) # (1, 62, 128)

for i in Xt_Train_3embed:
    print(i.shape)
print(type(Xt_Train_3embed[0]))



XXt_Train_3embed=[]

for i in Xt_Train_3embed:
    result = i[:, 0, :]
    print(result.shape)
    #c=i.astype(object)
    #result=np.squeeze(result)
    #print(result.shape)
    XXt_Train_3embed.append(result)

XXt_Train_3embed = np.array(XXt_Train_3embed)

##########################################   Yall  #################

Y_Train_labels_list = pd.read_pickle("YallDs1.pkl")

print("Y_______________all[0]")
print(Y_Train_labels_list[0])


print(type(Y_Train_labels_list))
print(type(Y_Train_labels_list[0]))

#Y_Train_labels_list = np.array(Y_Train_labels_list1)

encoder = LabelEncoder()
encoder.fit(Y_Train_labels_list)
encoded_Y = encoder.transform(Y_Train_labels_list)

true_Adad_Y_my_Train = encoded_Y[0:230]
true_Adad_Yt_my_Test = encoded_Y[230:334]


Y_all = np_utils.to_categorical(encoded_Y)# one-hot-encodeing

Y_my_Train = Y_all[0:230]
Yt_my_Test = Y_all[230:334]


print("Y-Train")
print(Y_my_Train[0])
print(Y_my_Train.shape)
print(type(Y_my_Train))
print(Y_my_Train[0].shape)
print(type(Y_my_Train[0]))


print("Y-Test")
print(Yt_my_Test[0])
print(Yt_my_Test.shape)
print(type(Yt_my_Test))
print(Yt_my_Test[0].shape)
print(type(Yt_my_Test[0]))

"""
############################## Y_Train ############################

Y_Train_labels_list = pd.read_pickle("lis_Y_all_Train_Ds1_small.pkl")

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


############################## Y_Test ############################

Yt_Train_labels_list = pd.read_pickle("lis_Y_all_Test_Ds1_small.pkl")

print(type(Yt_Train_labels_list))
print(type(Yt_Train_labels_list[0]))

#Y_Train_labels_list = np.array(Y_Train_labels_list1)

encoder = LabelEncoder()
encoder.fit(Yt_Train_labels_list)
encoded_Y = encoder.transform(Yt_Train_labels_list)
Yt_my_Train = np_utils.to_categorical(encoded_Y)


print("Y-Test")
print(Yt_my_Train.shape)
print(type(Yt_my_Train))
print(Yt_my_Train[0].shape)
print(type(Yt_my_Train[0]))
"""

'''
X_Train_3embed=X_Train_3embed.reshape(-1,128)

print("Finalllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll")

print("X-Train")
print(X_Train_3embed.shape)
print(type(X_Train_3embed))
print(X_Train_3embed[2].shape) # (1, 62, 128)

for i in X_Train_3embed:
    print(i.shape)
print(type(X_Train_3embed[0]))

'''
first_input = Input(shape=(1, 128))
first_dense = Dense(128)(first_input)
output_layer = Dense(85, activation='softmax')(first_dense)
outputs = Flatten()(output_layer)

model = Model(inputs=first_input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(XX_Train_3embeding, Y_my_Train, epochs=55, batch_size=32)


####################################### Infer #################################


prediction_probas = model.predict(XXt_Train_3embed) # [

predictions = [np.argmax(pred) for pred in prediction_probas]

#cm = confusion_matrix(true_Adad_Yt_my_Test, predictions) #from sklearn.metrics
#print(cm)

#print(classification_report(true_Adad_Yt_my_Test, predictions, digits=3))

print("predictions")
print(predictions[0])
print(prediction_probas.shape)
print(prediction_probas[0])
print(prediction_probas[0].shape)
print(prediction_probas[1].shape)
print(prediction_probas[2].shape)

print("Yt_my_Test")
print(Yt_my_Test.shape)
print(Yt_my_Test[0])
print(Yt_my_Test[0].shape)
print(Yt_my_Test[1].shape)
print(Yt_my_Test[2].shape)

#print(prediction_probas.shape[0])
#print(prediction_probas.shape[1])


print(len(predictions))

tags=list(encoder.inverse_transform(predictions))

print(tags)

print(len(predictions))

#print(classification_report(true_Adad_Yt_my_Test, predictions, digits=3))
print("f1_score(true_Adad_Yt_my_Test, predictions)")

print(f1_score(true_Adad_Yt_my_Test, predictions, average=None))

print(f1_score(true_Adad_Yt_my_Test, predictions, average='macro'))
print(f1_score(true_Adad_Yt_my_Test, predictions, average='micro'))
print(f1_score(true_Adad_Yt_my_Test, predictions, average='weighted'))









import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import Input, Model
from keras.layers import Embedding, TimeDistributed, Bidirectional, LSTM, Concatenate, Dropout, Dense, Conv1D, \
    GlobalMaxPooling1D
from keras.models import Sequential
import numpy as np

#################### X ############################

X_Train_3embed = pd.read_pickle("X_Train_3embeding.pkl")

print(len(X_Train_3embed))

print(X_Train_3embed[0].shape)
print(type(X_Train_3embed[0].shape))
#print(X_Train_3embed[0][0][0][0])
#a=X_Train_3embed[0]
#print(X_Train_3embed[0])
#print(a[0][0][0])
#print(X_Train_3embed[0][0]) # -0.02347

### ************** Test

X_Test_3embed = pd.read_pickle("X_Test_3embeding.pkl")


#################### Y ############################

Y_Train_labels_list = pd.read_pickle("lis_Y_all_Train.pkl")
encoder = LabelEncoder()
encoder.fit(Y_Train_labels_list)
encoded_Y = encoder.transform(Y_Train_labels_list)
# convert integers to dummy variables (i.e. one hot encoded)
#print("num_classssssssssssssssssssssssssssss")
# num_class=encoder.classes_()
# print(num_class)
Y_my_Train = np_utils.to_categorical(encoded_Y)
#print(len(Y_my_Train))

#print(len(Y_Train_labels_list))

#print(Y_Train_labels_list[0].shape)

#print(type(Y_Train_labels_list))

print(Y_my_Train.shape[1])

### ************** Test

X_Test_labels_list = pd.read_pickle("lis_Y_all_Test.pkl")
encoder = LabelEncoder()
encoder.fit(Y_Train_labels_list)
encoded_Y = encoder.transform(Y_Train_labels_list)
# convert integers to dummy variables (i.e. one hot encoded)
#print("num_classssssssssssssssssssssssssssss")
# num_class=encoder.classes_()
# print(num_class)
Y_my_Test = np_utils.to_categorical(encoded_Y)


######################  Model

# First model
#first_input = Input(shape=(len(X_Train_3embed) ,640))
first_input = Input(shape=(640,))

first_dense = Dense(128)(first_input)

# Second model
#second_input = Input((10, ))
#second_dense = Dense(64)(second_input)

# Concatenate both
#merged = Concatenate()([first_dense, second_dense])
#output_layer = Dense(1)(merged)
#input_shape=(len(X), 200)
output_layer = Dense(83, activation='softmax')(first_dense)

model = Model(inputs=first_input, outputs=output_layer)

model.summary()

#model.compile(optimizer='sgd', loss='mse')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


history = model.fit((X_Train_3embed, Y_my_Train), epochs=2, batch_size=32)

tst_results = model.evaluate(X_Test_3embed, Y_my_Train, verbose=2)
print("Test LOSS:", tst_results[0])
print("Test Accuracy:", tst_results[1])

'''
mymodel = Sequential()
mymodel.add(Dense(128, activation="relu"))
#mymodel.add(Dense(128, activation="relu")(X_Train_3embed))
mymodel.add(Dense(83, activation='softmax'))


mymodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


mymodel.fit(X_Train_3embed, Y_my_Train, epochs=2, batch_size=32)


#prediction_probas = model.predict(X_test[1])
prediction_probas = [np.argmax(pred) for pred in (mymodel.predict(X_Test_3embed))]
print(prediction_probas)
#print(prediction_probas.shape[0])


print(len(prediction_probas))

tags=list(encoder.inverse_transform(prediction_probas))

print(tags)

print(len(prediction_probas))


a = np.ndarray.flatten(ftrs_from_old)
print(a.shape)

mymodel = Sequential()

mymodel.add(Dense(, activation="relu"), name="my_fc_rel")(model.layers[-5].output))



model = Sequential()
model.add(Embedding(n_most_common_words, embedding_size, input_length=X.shape[1]))
model.add(Bidirectional(LSTM(128, dropout=0.7, recurrent_dropout=0.7)))
model.add(Dense(dummy_y.shape[1], activation='softmax'))

my_fc_rel = TimeDistributed(Dense(self._my_fc_rel_dim, activation="relu"), name="my_fc_rel")(model.layers[-5].output)
my_cls_rel = TimeDistributed(Dense(85, activation="sigmoid"), name="my_cls_rel")
my_pred_rel = my_cls_rel(my_fc_rel)
preds2 = [my_pred_rel]
my_new_model = Model(inputs=model.layers[-5].output, outputs=preds2)  # creae model
my_new_model.summary()
print(len(my_new_model.layers))from keras import Input, Model
#'''
from keras import Input, Model
from keras.layers import Dense, Flatten
import numpy as np

# Dummy data
X_Train_3embed = np.random.randn(230, 1, 128)
Y_my_Train = np.random.randn(230, 83)

print(X_Train_3embed.shape)
print(type(X_Train_3embed))
print(X_Train_3embed[0].shape) # (1, 128)
print(type(X_Train_3embed[0]))




#model
first_input = Input(shape=(1, 128))
first_dense = Dense(128)(first_input)
output_layer = Dense(83, activation='softmax')(first_dense)
outputs = Flatten()(output_layer)

model = Model(inputs=first_input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_Train_3embed, Y_my_Train, epochs=2, batch_size=32)
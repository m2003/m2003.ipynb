import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test, y_test)=mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train_encoded =to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

import numpy as np
x_train_reshaped=np.reshape(x_train,(60000,784))
x_test_reshaped =np.reshape(x_test,(10000,784))
print('x_train_reshaped shape:', x_train_reshaped.shape)
print('x_test_reshaped shape:', x_test_reshaped.shape)

x_mean=np.mean(x_train_reshaped)
x_std=np.std(x_train_reshaped)
epsilon =1e-10
x_train_norm=(x_train_reshaped-x_mean)/(x_std+epsilon)
x_test_norm=(x_test_reshaped-x_mean)/(x_std+epsilon)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model= Sequential([
    Dense(128,activation='relu',input_shape=(784,)),
    Dense(128,activation ='relu'),
    Dense(10,activation='softmax')
])

model.compile(
optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

model.fit(x_train_norm, y_train_encoded, epochs=3)
loss, accuracy=model.evaluate(x_test_norm,y_test_encoded)
print('Test Set accuracy:',accuracy*100)

preds=model.predict(x_test_norm)
print('Shape of preds:',preds.shape)

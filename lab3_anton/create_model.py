import tensorflow as tf
K = tf.keras
L = K.layers
import numpy as np
from imim import *

def make_model():
    model = K.Sequential()

    
    model.add(L.Conv2D(16, 3, input_shape=(64, 64, 3), padding = 'same'))
    model.add(L.LeakyReLU(0.1))
    
    model.add(L.Conv2D(32, 3, padding = 'same'))
    model.add(L.LeakyReLU(0.1))
    
    model.add(L.MaxPooling2D(pool_size=(2, 2)))
    model.add(L.Dropout(0.25))
    
    model.add(L.Conv2D(32, 3, padding = 'same'))
    model.add(L.LeakyReLU(0.1))
    
    model.add(L.Conv2D(64, 3, padding = 'same'))
    model.add(L.LeakyReLU(0.1))
    
    model.add(L.MaxPooling2D(pool_size=(2, 2)))
    model.add(L.Dropout(0.25))
    
    model.add(L.Flatten())
    model.add(L.Dense(256))
    model.add(L.LeakyReLU(0.1))
    model.add(L.Dropout(0.5))
    
    model.add(L.Dense(2))
    model.add(L.Activation("softmax"))

    
    return model

model = make_model()
#model.summary()

model.compile(
    loss='categorical_crossentropy',  # we train 10-way classification
    optimizer=K.optimizers.Adamax(),  # for8 SGD
    metrics=['accuracy']  # report accuracy during training
)

model.fit(
    x_train2, y_train,  # prepared data
    batch_size=4,
    epochs=10,
    #validation_data=(x_test2, y_test2),
    shuffle=True,
    verbose=0,
    #initial_epoch=last_finished_epoch or 0
)

model.save("my_model")

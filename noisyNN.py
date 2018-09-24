# -*- coding: utf-8 -*-

# https://arxiv.org/pdf/1412.6596.pdf
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.749.1795&rep=rep1&type=pdf
# %%
import numpy as np
import pandas as pd
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Input
from keras.models import Model
from keras.activations import relu, softmax
from keras.initializers import glorot_uniform, Identity
from keras.constraints import non_neg
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import adam, sgd
from keras.losses import categorical_crossentropy
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# %%
def noisy_model(ipt_dims, cls_num ,lr=1e-3):
    ipt_layer = Input((ipt_dims, ))
    layer1 = Dense(32, activation='relu', kernel_initializer='glorot_normal', name='l1')(ipt_layer)
    layer2 = Dense(16, activation='relu', kernel_initializer='glorot_normal', name='l2')(layer1)
    out1 = Dense(cls_num, activation='softmax', kernel_initializer='glorot_normal', name='o1')(layer2)

    q_o = Dense(
        cls_num, 
        kernel_initializer='glorot_normal', kernel_regularizer=l2(), 
        use_bias=False, name='q1', kernel_constraint=non_neg())(out1)
    base_model = Model(inputs=[ipt_layer], outputs=[out1])
    whole_model = Model(inputs=[ipt_layer], outputs=[q_o])

    base_model.compile(optimizer=adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
    whole_model.compile(optimizer=adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])
    return base_model, whole_model


# %%
data = pd.read_csv('adult.csv')
y = pd.get_dummies(data["Income"]).values
x = pd.get_dummies(data.drop('Income', axis=1), columns=[
    "WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
    "Race", "Gender", "NativeCountry"]).values
# %%
data.describe()

# %%
n_dims = x.shape[1]
# %%
noise_matrix = np.array([[0.7, 0.2],[0.3, 0.8]])
noise_y = np.dot(y, noise_matrix)

# %%
# train model with noise data
x_train, x_test, ny_train, ny_test, y_train, y_test = train_test_split(x, noise_y, y, test_size=0.2)
x_train, x_val, ny_train, ny_val, y_train, y_val = train_test_split(x_train, ny_train, y_train, test_size=0.2)
b_model, n_model = noisy_model(n_dims, 2, 1e-5)
phase1_stop = EarlyStopping()
tb = TensorBoard(write_grads=True, write_images=True, log_dir='./logs', histogram_freq=1)
K.set_learning_phase(1)
b_model.fit(
    x_train, y_train, batch_size=32, epochs=100, verbose=2,
    validation_data=(x_val, y_val), callbacks=[phase1_stop], shuffle=True
)
n_model.fit(
    x_train, y_train, batch_size=32, epochs=20, verbose=2,
    validation_data=(x_val, y_val), shuffle=True
)
# %%
n_model.get_layer(name='q1').get_weights()
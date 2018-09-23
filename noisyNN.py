# -*- coding: utf-8 -*-
# %%
import numpy as np
import pandas as pd
import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Input
from keras.models import Model
from keras.activations import relu, softmax
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import adam, sgd
from keras.losses import categorical_crossentropy
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# %%
def ident_ini(shape):
    return K.eye(shape[0])

class Q_layer(Layer):

    def __init__(self, output_dim, 
                activation=None,
                 kernel_initializer=ident_ini,
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(Q_layer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=ident_ini,
                                      trainable=True)
        super(Q_layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        output = K.dot(x, self.kernel)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
# %%
def noisy_model(ipt_dims, cls_num):
    ipt_layer = Input((ipt_dims, ))
    layer1 = Dense(32, activation='relu')(ipt_layer)
    layer2 = Dense(16, activation='relu')(layer1)
    out1 = Dense(cls_num, activation='softmax')(layer2)

    q_o = Q_layer(cls_num, kernel_initializer=ident_ini, kernel_regularizer=l2)(out1)
    base_model = Model(inputs=[ipt_layer], outputs=[out1])
    whole_model = Model(inputs=[ipt_layer], outputs=[q_o])

    base_model.compile(optimizer=adam(lr=5e-4), loss=categorical_crossentropy, metrics=[categorical_crossentropy])
    whole_model.compile(optimizer=adam(lr=5e-4), loss=categorical_crossentropy, metrics=[categorical_crossentropy])
    return base_model, whole_model


# %%
data = pd.read_csv('adult.csv')
y = pd.get_dummies(data["Income"])
x = pd.get_dummies(data.drop('Income', axis=1), columns=[
    "WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
    "Race", "Gender", "NativeCountry"]).values
# %%
x.shape

# %%
y = pd.get_dummies(data.target)
x = data.data
n_dims = x.shape[1]
# %%
noise_matrix = np.array([[0.1, 0.1],[0.1, 0.1]])
noise_y = np.dot(y, noise_matrix)

# %%
# train model with noise data
x_train, x_test, ny_train, ny_test, y_train, y_test = train_test_split(x, noise_y, y, test_size=0.2)
x_train, x_val, ny_train, ny_val, y_train, y_val = train_test_split(x_train, ny_train, y_train, test_size=0.2)
b_model, n_model = noisy_model(n_dims, 2)
phase1_stop = EarlyStopping()
b_model.fit(
    x_train, ny_train, batch_size=8, epochs=100, verbose=2,
    validation_data=(x_val, ny_val), callbacks=[phase1_stop]
)
n_model.fit(
    x_train, ny_train, batch_size=8, epochs=20, verbose=2,
    validation_data=(x_val, ny_val)
)
# %%
x
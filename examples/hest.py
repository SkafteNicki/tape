# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:48:31 2019

@author: nsde
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Conv1D, Flatten, Reshape, Layer, Lambda, Conv2DTranspose
import numpy as np
from rinokeras.layers import Stack
from sacred import Ingredient

from tape.models import AbstractTapeModel
from tape.models import ModelBuilder

#%% Conv1D transpose layer
class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        self._model = Sequential()
        self._model.add(Lambda(lambda x: K.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:,0]))
        self._model.summary()
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)

#%%
def pad_up_to(tensor, max_in_dims, constant_values):
    s = tf.shape(tensor)
    paddings = [[0, tf.maximum(m-s[i], tf.constant(0))] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(tensor, paddings, 'CONSTANT', constant_values=constant_values)

#%% Model
hparams = Ingredient('my_hparams')

@hparams.config
def model_cfg():
    latent_size = 32
    max_seq_len = 10000
    
class MyModel(AbstractTapeModel):
    @hparams.capture
    def __init__(self, n_symbols, latent_size=32, max_seq_len=10000):
        
        self.latent_size = latent_size
        self.max_seq_len = max_seq_len
        
        super().__init__(n_symbols)
        
        self.input_embedding = Embedding(n_symbols, 128)
        
        enc = Stack()
        enc.add(Conv1D(filters=32, kernel_size=7, strides=1, dilation_rate=2, activation='relu'))
        enc.add(Conv1D(filters=64, kernel_size=5, strides=1, dilation_rate=2, activation='relu'))
        enc.add(Conv1D(filters=128, kernel_size=3, strides=1, dilation_rate=2, activation='relu'))
                    
        self.enc_mu = Stack()
        self.enc_mu.add(enc)
        self.enc_mu.add(Flatten())
        self.enc_mu.add(Dense(latent_size))
        
        self.enc_std = Stack()
        self.enc_std.add(enc)
        self.enc_std.add(Flatten())
        self.enc_std.add(Dense(latent_size, activation='softplus'))
        
        self.dec = Stack()
        self.dec.add(Dense(1000))
        self.dec.add(Reshape((100, 10)))
        self.dec.add(Conv1DTranspose(filters=128, kernel_size=3, strides=1, dilation_rate=2, activation='relu'))
        self.dec.add(Conv1DTranspose(filters=64, kernel_size=3, strides=1, dilation_rate=2, activation='relu'))
        self.dec.add(Conv1DTranspose(filters=32, kernel_size=3, strides=1, dilation_rate=2, activation='relu'))
        
    def call(self, inputs):
        sequence = inputs['primary']
        
        embedded = self.input_embedding(sequence)
        pad_embedded = pad_up_to(embedded, (-1, self.max_seq_len, -1), 0)
        pad_embedded.set_shape((None,self.max_seq_len,128))
        
        z_mu = self.enc_mu(pad_embedded)
        z_std = self.enc_std(pad_embedded)
        z = z_mu + K.random_normal(K.shape(z_std)) * z_std
        
        encoder_output = self.dec(z)
        
        inputs['encoder_output'] = encoder_output
        return inputs
        
        
    def get_optimal_batch_sizes(self):
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1300, 2000, 3000])
        batch_sizes = np.array([4, 4, 4, 4, 3, 3, 3, 2, 1, 0.5, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes

#%% Register
ModelBuilder.add_model('my_model', MyModel)

#%%
if __name__ == '__main__':   
    from tape.__main__ import proteins
    proteins.run_commandline()
        
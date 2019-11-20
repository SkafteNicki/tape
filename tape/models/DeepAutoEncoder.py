from typing import Optional, Tuple, List

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda
import numpy as np
from sacred import Ingredient

import rinokeras as rk
from rinokeras.layers import Stack, ResidualBlock, PositionEmbedding, PaddedConv
from tensorflow.keras.layers import Conv1D, Cropping1D, BatchNormalization, AveragePooling1D, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape

from .AbstractTapeModel import AbstractTapeModel

dae_hparams = Ingredient('deepautoencoder')

@dae_hparams.config
def configure_ae():
    n_layers = 5 # actual layers are 2 * 6 * n_layers
    n_filters = 256
    kernel_size = 9
    latent_size = 1000
    pooling_type = 'avarage'
    dropout = 0

class DeepAutoEncoder(AbstractTapeModel):
    
    @dae_hparams.capture
    def __init__(self, n_symbols, n_layers=5, length=3000, latent_size=1000, n_filters=256,
                 kernel_size=5, pooling_type='average', dropout=0):
        super().__init__(n_symbols)
        self._n_layers = n_layers
        self._length = length
        self._latent_size = latent_size
        self._kernel_size = kernel_size
        self._n_filters = n_filters
        pool = AveragePooling1D if pooling_type=='average' else MaxPooling1D
        
        input_embedding = Stack()
        input_embedding.add(Embedding(n_symbols, 128, input_length=self._length))
        input_embedding.add(Lambda(lambda x: x * np.sqrt(n_filters)))
        input_embedding.add(PositionEmbedding())
        input_embedding.add(PaddedConv(1, n_filters, kernel_size, 1, activation='relu', dropout=dropout))
        
        encoder = Stack()
        encoder.add(input_embedding)
        for _ in range(6):
            for _ in range(n_layers):
                encoder.add(ResidualBlock(1, n_filters, kernel_size, activation='relu', dilation_rate=1, dropout=dropout))
            encoder.add(pool(2,2))
        
        latent = Stack()
        latent.add(Flatten())
        latent.add(Dense(self._latent_size))
        
        decoder = Stack()
        decoder.add(Dense(47*n_filters, input_shape=(self._latent_size,), activation='relu'))
        decoder.add(Reshape((47, n_filters)))
        for _ in range(6):
            decoder.add(UpSampling1D(2))
            for _ in range(n_layers):
                encoder.add(ResidualBlock(1, n_filters, kernel_size, activation='relu', dilation_rate=1, dropout=dropout))
        decoder.add(Cropping1D((0,8)))

        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent

    def call(self, inputs):
        sequence = inputs['primary']
        L = tf.shape(sequence)[1]
        pad_sequence = tf.pad(sequence, [[0,0], [0, self._length-L]], 'CONSTANT', constant_values=0)
        pad_sequence.set_shape((None, self._length))        
        encoder_output = self.encoder(pad_sequence)
        latent = self.latent(encoder_output)
        decoder_output = self.decoder(latent)
        inputs['global_emb'] = latent
        inputs['encoder_output'] = decoder_output[:,:L]
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1300, 2000, 3000])
        batch_sizes = np.array([4, 4, 4, 4, 3, 3, 3, 2, 1, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes
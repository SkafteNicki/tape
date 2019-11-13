from typing import Optional, Tuple, List

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda
import numpy as np
from sacred import Ingredient

import rinokeras as rk
from rinokeras.layers import Stack
from tensorflow.keras.layers import Conv1D, Cropping1D, BatchNormalization, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape

from tape.models import AbstractTapeModel

ae_hparams = Ingredient('ae')

@ae_hparams.config
def configure_vae():
    n_filters = 256
    kernel_size = 5
    latent_size = 1000


class MyAE(AbstractTapeModel):
    @ae_hparams.capture
    def __init__(self, n_symbols, length=3000, latent_size=1000, n_filters=256,
                 kernel_size=5):
        super().__init__(n_symbols)
        self._length = length
        self._latent_size = latent_size
        self._kernel_size = kernel_size
        self._n_filters = n_filters
        encoder = Stack()
        encoder.add(Embedding(n_symbols, 128, input_length=self._length))
        encoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Flatten())
        encoder.add(Dense(self._latent_size))
        
        decoder = Stack()
        decoder.add(Dense(47*n_filters, input_shape=(self._latent_size,), activation='relu'))
        decoder.add(Reshape((47, n_filters)))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(n_filters, kernel_size, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(Cropping1D((0,8)))

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        sequence = inputs['primary']
        L = tf.shape(sequence)[1]
        protein_length = inputs['protein_length']
        pad_sequence = tf.pad(sequence, [[0,0], [0, self._length-L]], 'CONSTANT', constant_values=0)
        pad_sequence.set_shape((None, self._length))
        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(pad_sequence, protein_length)
        encoder_output = self.encoder(pad_sequence, mask=sequence_mask)
        decoder_output = self.decoder(encoder_output, mask=sequence_mask)
        inputs['global_emb'] = encoder_output
        inputs['encoder_output'] = decoder_output[:,:L]
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1300, 2000, 3000])
        batch_sizes = np.array([4, 4, 4, 4, 3, 3, 3, 2, 1, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes
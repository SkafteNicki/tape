from typing import Optional, Tuple, List

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda
import numpy as np
from sacred import Ingredient

import rinokeras as rk
from rinokeras.layers import Stack, ResidualBlock, PaddedConv, PositionEmbedding
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Conv1D, Dropout, BatchNormalization, Layer, LeakyReLU, Conv2DTranspose, Cropping1D, BatchNormalization, MaxPooling1D, UpSampling1D, Flatten, Dense, Reshape

from tape.models import AbstractTapeModel
from tape.models import ModelBuilder

class MyRes(AbstractTapeModel):
    def __init__(self,
                 n_symbols: int,
                 n_layers: int = 35,
                 filters: int = 256,
                 kernel_size: int = 9,
                 layer_norm: bool = True,
                 activation: str = 'elu',
                 dilation_rate: int = 2,
                 dropout: Optional[float] = 0.1) -> None:
        super().__init__(n_symbols)
        self.n_symbols = n_symbols
        self.n_layers = n_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.layer_norm = layer_norm
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        print(self)

        input_embedding = Stack()
        input_embedding.add(Embedding(n_symbols, 128))
        input_embedding.add(Lambda(lambda x: x * np.sqrt(filters)))
        input_embedding.add(PositionEmbedding())

        encoder = Stack()
        encoder.add(input_embedding)
        encoder.add(PaddedConv(1, filters, kernel_size, 1, activation, dropout))
        encoder.add(ResidualBlock(1, filters, kernel_size, activation=activation,
                                  dilation_rate=1, dropout=dropout))
        for layer in range(n_layers - 1):
            encoder.add(ResidualBlock(1, filters, kernel_size, activation=activation,
                                      dilation_rate=dilation_rate, dropout=dropout,
                                      add_checkpoint=layer % 5 == 0))

        self.encoder = encoder
        
        self.z_mu = PaddedConv(1, 4, kernel_size, 1, 'linear', 0.0)
        self.z_var = PaddedConv(1, 4, kernel_size, 1, 'linear', 0.0)
        
        decoder = Stack()
        decoder.add(PaddedConv(1, filters, kernel_size, 1, activation, dropout))
        decoder.add(ResidualBlock(1, filters, kernel_size, activation=activation,
                                  dilation_rate=1, dropout=dropout))
        for layer in range(n_layers - 1):
            decoder.add(ResidualBlock(1, filters, kernel_size, activation=activation,
                                      dilation_rate=dilation_rate, dropout=dropout,
                                      add_checkpoint=layer % 5 == 0))

        self.decoder = decoder

    def __str__(self) -> str:
        outstr = []
        outstr.append('Resnet with Parameters:')
        outstr.append(f'\tn_layers: {self.n_layers}')
        outstr.append(f'\tfilters: {self.filters}')
        outstr.append(f'\tkernel_size: {self.kernel_size}')
        outstr.append(f'\tactivation: {self.activation}')
        outstr.append(f'\tdilation_rate: {self.dilation_rate}')
        outstr.append(f'\tdropout: {self.dropout}')
        return '\n'.join(outstr)

    def call(self, inputs):
        """
        Args:
            sequence: tf.Tensor[int32] - Amino acid sequence,
                a padded tensor with shape [batch_size, MAX_PROTEIN_LENGTH]
            protein_length: tf.Tensor[int32] - Length of each protein in the sequence, a tensor with shape [batch_size]

        Output:
            encoder_output: tf.Tensor[float32] - embedding of each amino acid
                a tensor with shape [batch_size, MAX_PROTEIN_LENGTH, filters]
        """

        sequence = inputs['primary']
        protein_length = inputs['protein_length']

        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(sequence, protein_length)
        encoder_output = self.encoder(sequence, mask=sequence_mask)
        
        z_mu = self.z_mu(encoder_output, mask=sequence_mask)
        z_var = tf.nn.softplus(self.z_var(encoder_output, mask=sequence_mask)) + 1e-4
        z = z_mu + tf.random_normal(tf.shape(z_var)) * tf.sqrt(z_var)

        decoder_output = self.decoder(z, mask=sequence_mask)
        
        inputs['z_mu'] = z_mu
        inputs['z_var'] = z_var
        inputs['encoder_output'] = decoder_output
        z_concat = tf.reshape(z, (tf.shape(z)[0], -1))
        z_concat = tf.pad(z_concat, [[0, 0], [0, 5000-tf.shape(z_concat)[1]]], 'CONSTANT', constant_values=-100)
        z_concat.set_shape((None, 5000))
        inputs['global_emb'] = z_concat
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1300, 2000, 3000])
        batch_sizes = np.array([1.5, 1.5, 1.5, 1.5, 1, 1, 1, 0, 0, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes

# Register the model
ModelBuilder.add_model('myres', MyRes)

class MyAE(AbstractTapeModel):
    def __init__(self, n_symbols, length=3000):
        super().__init__(n_symbols)
        self._length = length
        
        encoder = Stack()
        encoder.add(Embedding(n_symbols, 128, input_length=self._length))
        encoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling1D(2,2))
        encoder.add(Flatten())
        encoder.add(Dense(1000))
        
        decoder = Stack()
        decoder.add(Dense(47*256, input_shape=(1000,), activation='relu'))
        decoder.add(Reshape((47, 256)))
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
        decoder.add(BatchNormalization())
        decoder.add(UpSampling1D(2))
        decoder.add(Conv1D(256, 5, strides=1, padding='same', dilation_rate=1, activation='relu'))
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
        print('pad_seq', pad_sequence)
        encoder_output = self.encoder(pad_sequence, mask=sequence_mask)
        decoder_output = self.decoder(encoder_output, mask=sequence_mask)
        print('enc_out', encoder_output)
        print('dec_out', decoder_output)
        inputs['global_emb'] = encoder_output
        inputs['encoder_output'] = decoder_output[:,:L]
        return inputs

    def get_optimal_batch_sizes(self) -> Tuple[List[int], List[int]]:
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1300, 2000, 3000])
        batch_sizes = np.array([4, 4, 4, 4, 3, 3, 3, 2, 1, 0, 0])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes

        
ModelBuilder.add_model('myae', MyAE)

if __name__ == '__main__':
    from tape.__main__ import proteins
    proteins.run_commandline()
#    from tape.run_eval import main
#    main()

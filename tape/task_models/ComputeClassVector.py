import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout

import rinokeras as rk
from rinokeras.layers import WeightNormDense as Dense
from rinokeras.layers import Stack, LayerNorm, ApplyAttentionMask


class ComputeClassVector(Model):

    def __init__(self,
                 input_name: str = 'encoder_output',
                 output_name: str = 'cls_vector',
                 mean_type: str = 'soft'):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name
        self.compute_attention = Stack([LayerNorm(), Dense(1, activation='linear'), Dropout(0.1)])
        self.attention_mask = ApplyAttentionMask()
        self._mean_type = mean_type
        
    def call(self, inputs):
        sequence_mask = rk.utils.convert_sequence_length_to_sequence_mask(
                inputs['primary'], inputs['protein_length'])
        if self._mean_type != 'normal':
            
    
            encoder_output = inputs[self._input_name]
            attention_weight = self.compute_attention(encoder_output)
    
            attention_weight = self.attention_mask(
                attention_weight, mask=sequence_mask[:, :, None])
            
            attention = tf.nn.softmax(attention_weight, 1)
            if self._mean_type == 'hard':
                print(encoder_output)
                print(attention)
                idx = [tf.range(tf.shape(encoder_output)[0]),
                       tf.cast(tf.squeeze(tf.argmax(attention, axis=1)), tf.int32)]
                cls_vector = tf.gather_nd(encoder_output, tf.transpose(idx))
            else:
                cls_vector = tf.squeeze(tf.matmul(encoder_output, attention, transpose_a=True), 2)
            inputs[self._output_name] = cls_vector
        else:
            sequence_mask = tf.cast(sequence_mask, tf.float32)
            inputs[self._output_name] = tf.reduce_sum(inputs[self._input_name] * sequence_mask[:,:,None], axis=1) / tf.reduce_sum(sequence_mask, axis=1, keepdims=True)

        return inputs

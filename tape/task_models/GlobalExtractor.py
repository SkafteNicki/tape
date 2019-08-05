# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:24:30 2019

@author: nsde
"""

from tensorflow.keras import Model
class GlobalExtractor(Model):
    def __init__(self,
                 input_name: str = 'encoder_output',
                 output_name: str = 'cls_vector'):
        super().__init__()
        self._input_name = input_name
        self._output_name = output_name

    def call(self, inputs):
        inputs[self._output_name] = inputs['global_emb']
        return inputs
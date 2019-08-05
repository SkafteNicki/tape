# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:36:32 2019

@author: nsde
"""

from typing import List, Dict, Tuple

import tensorflow as tf
from sacred import Ingredient

from tape.data_utils import PFAM_VOCAB, deserialize_pfam_sequence
from tape.task_models import BidirectionalOutputShift, AminoAcidClassPredictor,RandomSequenceMask
from .AbstractLanguageModelingTask import AbstractLanguageModelingTask

#%%
class OwnLanguageModelingTaskNext(AbstractLanguageModelingTask):
    def __init__(self):
        n_symbols = len(PFAM_VOCAB)
        super().__init__(
            key_metric='LMACC',
            deserialization_func=deserialize_pfam_sequence,
            n_classes=n_symbols,
            label_name='primary',
            input_name='encoder_output',
            output_name='lm_logits')

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        loss, metrics = super().loss_function(inputs, outputs)
        loss -= inputs['kl_scaled']
        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.append(BidirectionalOutputShift(self._input_name, 'shifted_logits'))
        layers.append(AminoAcidClassPredictor(self._n_classes, 'shifted_logits', self._output_name, use_conv=False))
        return layers

#%%
mask_params = Ingredient('mask')

@mask_params.config
def mask_config():
    percentage = 0.15  # noqa: F841
    style = 'random'  # noqa: F841

class OwnLanguageModelingTaskMask(AbstractLanguageModelingTask):
    @mask_params.capture
    def __init__(self,
                 percentage: float = 0.15,
                 style: str = 'random'):
        n_symbols = len(PFAM_VOCAB)
        mask_token = PFAM_VOCAB['<MASK>']

        super().__init__(
            key_metric='BERTAcc',
            deserialization_func=deserialize_pfam_sequence,
            n_classes=n_symbols,
            label_name='original_sequence',
            input_name='encoder_output',
            output_name='bert_logits',
            mask_name='bert_mask')
        self._mask_token = mask_token
        self._percentage = percentage
        self._style = style

    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        loss, metrics = super().loss_function(inputs, outputs)
        loss -= inputs['kl_scaled']
        return loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.insert(0, RandomSequenceMask(self._n_classes, self._mask_token, self._percentage, self._style))
        layers.append(AminoAcidClassPredictor(self._n_classes, self._input_name, self._output_name, use_conv=True))
        return layers

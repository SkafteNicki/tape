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
elbo_params = Ingredient('elbo')

@elbo_params.config
def mask_config():
    percentage = 0.15  # noqa: F841
    style = 'random'  # noqa: F841
    beta = 1.0
    warmup = 1
    
class VariationalObjective(AbstractLanguageModelingTask):

    @elbo_params.capture
    def __init__(self,
                 percentage: float = 0.15,
                 style: str = 'random',
                 beta: float = 1.0,
                 warmup: int = 1):
        n_symbols = len(PFAM_VOCAB)
        mask_token = PFAM_VOCAB['<MASK>']
        mask_name = 'bert_mask' if percentage != 0 else 'sequence_mask'
        super().__init__(
            key_metric='BERTAcc',
            deserialization_func=deserialize_pfam_sequence,
            n_classes=n_symbols,
            label_name='original_sequence',
            input_name='encoder_output',
            output_name='bert_logits',
            mask_name=mask_name)
        self._mask_token = mask_token
        self._percentage = percentage
        self._style = style
        self._beta_scale = beta
        self.beta = tf.Variable(0.0)
        self.inc_beta = tf.assign_add(self.beta, 1-1/warmup)
        
    def loss_function(self,
                      inputs: Dict[str, tf.Tensor],
                      outputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        loss, metrics = super().loss_function(inputs, outputs)
        metrics['recon_loss'] = loss
        kl = -0.5 * tf.reduce_mean(1 + tf.log(outputs['z_var']) - outputs['z_mu']**2 - outputs['z_var'])
        
        with tf.control_dependencies(self.inc_beta):
            beta = self._beta_scale * tf.minimum(1.0, tf.identity(self.beta))
        
        total_loss = loss + beta * kl
        metrics['kl'] = kl
        metrics['z_var'] = tf.reduce_mean(outputs['z_var'])
        return total_loss, metrics

    def build_output_model(self, layers: List[tf.keras.Model]) -> List[tf.keras.Model]:
        layers.insert(0, RandomSequenceMask(self._n_classes, self._mask_token, self._percentage, self._style))
        layers.append(AminoAcidClassPredictor(self._n_classes, self._input_name, self._output_name, use_conv=True))
        return layers

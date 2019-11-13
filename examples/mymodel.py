#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:33:21 2019

@author: nsde
"""

import numpy as np
from tensorflow.keras.layers import Embedding
from sacred import Ingredient

from tape.models import AbstractTapeModel
from tape.models import ModelBuilder

class Word2Vec(AbstractTapeModel):
    def __init__(self, n_symbols):
        super().__init__(n_symbols)
        self.embedding = Embedding(n_symbols, 1024)

    def call(self, inputs):
        sequence = inputs['primary']
        inputs['encoder_output'] = self.embedding(sequence)
        return inputs

    def get_optimal_batch_sizes(self):
        bucket_sizes = np.array([100, 200, 300, 400, 600, 900, 1000, 1200, 1300, 2000, 3000])
        batch_sizes = np.array([10, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5])

        batch_sizes = np.asarray(batch_sizes * self._get_gpu_memory(), np.int32)
        batch_sizes[batch_sizes <= 0] = 1
        return bucket_sizes, batch_sizes

ModelBuilder.add_model('w2v', Word2Vec)

from tape.analysis import get_config
from tape.__main__ import proteins
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir')
    parser.add_argument('--datafile', default='')
    args = parser.parse_args()

    config = get_config(args.outdir)
    task = config['tasks']

    if isinstance(task, (tuple, list)):
        task = task[0]

    config_updates = {
        'tasks': task,
        'load_task_from': os.path.join(args.outdir, 'task_weights.h5'),
        'save_outputs': True,
        'datafile': args.datafile}

    proteins.run(
        'eval',
        named_configs=[os.path.join(args.outdir, '1', 'config.json')],
        config_updates=config_updates,
    )


if __name__ == '__main__':
    main()

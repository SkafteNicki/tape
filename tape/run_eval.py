from tape.analysis import get_config
import os

from tape.__main__ import proteins

import argparse
import numpy as np
import pickle as pkl
from scipy.stats import spearmanr

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
    print(config)
    if config['tasks'] == 'stability':
        with open(os.path.join(args.outdir, 'outputs.pkl'), 'rb') as f:
            results = pkl.load(f)
            predictions = np.array(results['prediction'])
            true_values = np.array(results['stability_score'])
            score, _ = spearmanr(true_values, predictions)
            
        print('stability score:', score)
        
    if config_updates['tasks'] == 'fluorescence':
        def postprocess(data):
            """ Converts list of numpy arrays to flat numpy array. """
            _clean = [float(i) for i in data]
            clean = np.array(_clean)
            return clean
        
        with open(os.path.join(args.outdir, 'outputs.pkl'), 'rb') as f:
            results = pkl.load(f) 
            
            predictions = postprocess(results['prediction'])
            true_values = postprocess(results['log_fluorescence'])
            
            score, _ = spearmanr(true_values, predictions)
            
        print('fluorescence score:', score)

if __name__ == '__main__':
    main()
    
    
            

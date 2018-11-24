import os
import sys

sys.path.append('..')

import numpy as np

from hparams import hparams
from datasets.audio import save_wavernn_wav


def main():
    os.chdir('../')

    # Get path
    cpp_dir = os.path.join('cpp')
    input_path = os.path.join(cpp_dir, 'output.txt')
    output_path = os.path.join(cpp_dir, 'output.wav')
    os.makedirs(cpp_dir, exist_ok=True)

    print(f'Loading input from {input_path}')

    samples = np.loadtxt(input_path, delimiter='\n')
    save_wavernn_wav(samples, output_path, hparams.sample_rate)

    print(f'Saving output into {output_path}')
    print('Finish!!!')


if __name__ == '__main__':
    main()

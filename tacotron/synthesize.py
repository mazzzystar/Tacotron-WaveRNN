import argparse
import os
import re
import time
from time import sleep

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm


def generate_fast(model, text):
    model.synthesize(text, None, None, None, None)


def run_live(args, checkpoint_path, hparams):
    # Log to Terminal without keeping any records in files
    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    # Generate fast greeting message
    greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
    log(greetings)
    generate_fast(synth, greetings)

    # Interaction loop
    while True:
        try:
            text = input()
            generate_fast(synth, text)

        except KeyboardInterrupt:
            leave = 'Thank you for testing our features. see you soon.'
            log(leave)
            generate_fast(synth, leave)
            sleep(2)
            break


def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, text in enumerate(tqdm(sentences)):
            start = time.time()
            mel_filename, speaker_id = synth.synthesize([text], [i + 1], eval_dir, log_dir, None)

            file.write('{}|{}|{}\n'.format(text, mel_filename[0], speaker_id[0]))

    log('synthesized mel spectrograms at {}'.format(eval_dir))


def run_synthesis(args, checkpoint_path, input_dir, output_dir, hparams):
    if args.GTA:
        synth_dir = os.path.join(output_dir, 'gta')

        # Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)
    else:
        synth_dir = os.path.join(output_dir, 'natural')

        # Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)

    metadata_filename = os.path.join(input_dir, 'train.txt')

    log(hparams_debug_string())

    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, gta=GTA)
    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        frame_shift_ms = hparams.hop_size / hparams.sample_rate
        hours = sum([int(x[4]) for x in metadata]) * frame_shift_ms / (3600)
        log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

    metadata = [metadata[i: i + hparams.tacotron_synthesis_batch_size] for i in range(0, len(metadata), hparams.tacotron_synthesis_batch_size)]

    log('starting synthesis')
    mel_dir = os.path.join(input_dir, 'mels')
    wav_dir = os.path.join(input_dir, 'audio')
    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i, meta in enumerate(tqdm(metadata)):
            texts = [m[5] for m in meta]
            mel_filenames = [os.path.join(mel_dir, m[1]) for m in meta]
            wav_filenames = [os.path.join(wav_dir, m[0]) for m in meta]
            basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
            mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, None, mel_filenames)

            for elems in zip(wav_filenames, mel_filenames, mel_output_filenames, speaker_ids, texts):
                file.write('|'.join([str(x) for x in elems]) + '\n')

    log('synthesized mel spectrograms at {}'.format(synth_dir))


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
    input_dir = os.path.join(args.base_dir, 'training_data')
    output_dir = os.path.join(args.base_dir, 'tacotron_output')

    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

    if args.mode == 'eval':
        run_eval(args, checkpoint_path, output_dir, hparams, sentences)
    elif args.mode == 'synthesis':
        run_synthesis(args, checkpoint_path, input_dir, output_dir, hparams)
    else:
        run_live(args, checkpoint_path, hparams)

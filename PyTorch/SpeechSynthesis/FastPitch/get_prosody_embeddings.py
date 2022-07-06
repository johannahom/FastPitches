# Antti: Extraction of reference encoder embeddings of the training data 
# adjust checkpoints, paths, and mel-spectrum extraction params

#import matplotlib
#import matplotlib.pylab as plt
import torch
import argparse
import models
import time
import sys, os
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
from common import utils, layers
from common.utils import load_wav_to_torch
from csv import DictReader
import csv

device = "cuda:0"
stft = layers.TacotronSTFT()


def parse_args(parser):
    # parser.add_argument('--fastpitch', type=str, default='prob_out2/FastPitch_checkpoint_400.pt',
    parser.add_argument('--fastpitch', type=str, default='/disk/scratch/s2132904/slt_2022/FastPitches/PyTorch/SpeechSynthesis/FastPitch/referemce_encoder/FastPitch_checkpoint_200.pt',
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')

    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=1)

    text_processing = parser.add_argument_group('Text processing parameters')
    text_processing.add_argument('--text-cleaners', nargs='*',
                                 default=['english_cleaners_v2'], type=str,
                                 help='Type of text cleaners for input text')
    text_processing.add_argument('--symbol-set', type=str, default='english_basic',
                                 help='Define symbol set for input text')

    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')

    cond = parser.add_argument_group('conditioning on additional attributes')
    cond.add_argument('--n-speakers', type=int, default=18993,
                      help='Number of speakers in the model.') #18993
    cond.add_argument('--speaker-independent', action='store_true',
                      help='Add speaker conditioning after variance adapters')
    cond.add_argument('--cwt-accent', action='store_true',
                      help='Enable CWT Accent Conditioning')
    cond.add_argument('--reference-encoder', action='store_true',
                      help='Enable Prosodic Embedding Conditioning')
    cond.add_argument('--mels-downsampled', action='store_true',
                      help='Using downsampled mels for reference encoder') #Can be used for
    return parser



def load_model_from_ckpt(checkpoint_path, ema, model):


    checkpoint_data = torch.load(checkpoint_path,map_location = device)
    status = ''

    if 'state_dict' in checkpoint_data:
        sd = checkpoint_data['state_dict']
        if ema and 'ema_state_dict' in checkpoint_data:
            sd = checkpoint_data['ema_state_dict']
            status += ' (EMA)'
        elif ema and not 'ema_state_dict' in checkpoint_data:
            print(f'WARNING: EMA weights missing for {checkpoint_data}')

        if any(key.startswith('module.') for key in sd):
            sd = {k.replace('module.', ''): v for k,v in sd.items()}
        status += ' ' + str(model.load_state_dict(sd, strict=False))
    else:
        model = checkpoint_data['model']
    print(f'Loaded {checkpoint_path}{status}')
    return model


def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):

    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    setattr(model_args, "energy_conditioning",True)
    model_config = models.get_model_config(model_name, model_args)
   
    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)
    
    if checkpoint is not None:
        model = load_model_from_ckpt(checkpoint, ema, model)
 
    model.eval()
   
    return model.to(device)


def wav2mel(filename):
              
    
    audio, sampling_rate = load_wav_to_torch(filename)
    audio_norm = audio / 32768.0 #hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)

    return melspec


parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)

parser = parse_args(parser)
args, unk_args = parser.parse_known_args()
generator = load_and_setup_model(
            'FastPitch', parser, args.fastpitch, args.amp, device,
            unk_args=unk_args, forward_is_infer=True, ema=args.ema,
            jitable=args.torchscript)



training_files = "/disk/scratch2/jomahony/Spotify-Question-Answers/reference_encoder/pretraining_refencoder_18993_train.txt"
#ref_list = open(training_files, "r")
#ref_list = ref_list.readlines()

all_embeddings = []
all_info = []
#for l in ref_list:
#    print("."),
#    spkr = None
#    cols = l.split("|")
#    if len(cols) == 4:
        
#        wav, pitch, text, spkr = cols
        
#    elif len(cols) == 3:
        
#        wav, pitch, text = cols #l.split("|")
#    else:
#        wav, pitch = cols

fpaths_and_text = []
with open(training_files, encoding='utf-8') as f:
    dict_reader = DictReader(f, delimiter='|', quoting=csv.QUOTE_NONE)
    fpaths_and_text = list(dict_reader)

for f in fpaths_and_text:
    mel_path = f['mels']
    #qa = mel_path.rstrip(".pt").split("_")[-2]
    #print(qa)
    speaker = f['speaker']
    
    mel = torch.load(f['mels_ds']).unsqueeze(0)
    #print(mel)   
    length = torch.tensor([mel.shape[-1]])

    with torch.no_grad():
        embeddings = generator.reference_encoder(mel.cuda(), length) #.unsqueeze(0))
 
    all_embeddings.append(embeddings.detach().cpu().numpy()[0])
    all_info.append((mel_path, speaker))

with open('labels.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['mel_path', 'speaker'])
    for row in all_info:
        csv_out.writerow(row)

    
all_embeddings = np.array(all_embeddings)
np.savetxt("all_embeddings.txt", all_embeddings, fmt="%.5f", delimiter="\t")



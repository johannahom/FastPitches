import numpy as np
import torch
import math
import librosa 
import torch.nn.functional as F

def estimate_energy(mel, norm=True, log=True):

    energy = torch.norm(mel.float(), dim=0, p=2)
    if log:
        energy = normalize_energy(energy)
    if norm:
        energy = z_normalise_values(energy)
    return energy

def normalize_energy(energy_values): # @Johannah, we might want to be able to specify values here via the input file
    '''Converts values to log domain '''

    def get_log(x):
        return 10 * (math.log10(x))

    energy_values = energy_values.numpy()
    energy_values[energy_values == 0] = np.nan
    le = np.vectorize(get_log)
    log_energy = le(energy_values)
    log_energy = np.where(np.isnan(log_energy), 0.0, log_energy)
    log_energy = torch.from_numpy(log_energy)

    return log_energy


def estimate_pitch(wav, mel_len, method='pyin', two_pass_method=False, normalize_mean=None,
                   normalize_std=None, norm_method='default', pitch_norm=True, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':
        snd, sr = librosa.load(wav)
        pitch_mel = pyin_pitch_extraction(snd, mel_len)   
 
        if two_pass_method == True:
            pfloor, pceiling = get_speaker_limits(pitch_mel)
            pitch_mel = pyin_pitch_extraction(snd, mel_len, pfloor=pfloor, pceiling=pceiling)
    else:
        raise ValueError

    pitch_mel = pitch_mel.float() 

    print(pitch_mel)
    print(pitch_mel.size)       
    if pitch_norm is True:
        if norm_method == 'zscore':
            pitch_mel = z_normalise_values(pitch_mel)
        elif norm_method == 'semitones':
            pitch_mel = normalize_pitch_st(pitch_mel)
        else:
            pitch_mel = normalize_pitch_default(pitch_mel, normalize_mean, normalize_std)

        return pitch_mel

    else:
        return pitch_mel #shape = (1,<mel_len>)


def normalize_pitch_default(pitch, mean, std):

    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0

    return pitch


def z_normalise_values(values):
    
    def get_zscore(x, mean, stf):
        return (x-mean)/std
         
    values = values.numpy()
    values[values == 0] = np.nan
    mean = np.nanmean(values)
    std = np.nanstd(values)
    print(mean, std)
    zscore = np.vectorize(get_zscore)
    zscores = zscore(values, mean, std)
    zscores = np.where(np.isnan(zscores), 0.0, zscores)
    zscores = torch.from_numpy(zscores)
    
    return zscores


def get_speaker_limits(pitch_values):
    ''' Calculate speaker specific pitch
    floor and ceiling based on DeLooze method'''
    pitch_values = pitch_values[pitch_values != 0.0]    
    q35 = np.nanquantile(pitch_values, .35)
    q65 = np.nanquantile(pitch_values, .65)
    pfloor = q35 * 0.72 - 10 
    pceiling = q65 * 1.9 + 10

    return pfloor, pceiling 


def normalize_pitch_st(pitch_values): # @Johannah, we might want to be able to specify values here via the input file
    '''Converts values to semitones based
    on speaker utterance median '''

    def get_semitones(x, speaker_median):
        return 12 * (math.log2(x/speaker_median))

    pitch_values = pitch_values.numpy()
    pitch_values[pitch_values == 0] = np.nan
    speaker_median = np.nanmean(pitch_values)
    st = np.vectorize(get_semitones)
    semitones = st(pitch_values, speaker_median)
    semitones = np.where(np.isnan(semitones), 0.0, semitones)
    semitones = torch.from_numpy(semitones)
    #print(semitones.shape)
    return semitones


def pyin_pitch_extraction(snd, mel_len, pfloor=75, pceiling=600, n_formants=1):
    
    #need statement incase floor or ceiling goes wrong
    if not pfloor or pceiling == np.nan:
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=pfloor,
            fmax=pceiling, frame_length=1024)
    else:
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=65,
            fmax=2093, frame_length=1024) #changed from 600..not sure why the pyin default is so high??
    assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

    pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
    pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
    pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))
    #print(pitch_mel.shape)
    if n_formants > 1:
        raise NotImplementedError
    
    return pitch_mel

# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import functools
import json
import re
from pathlib import Path
import tgt
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom
import torch.nn.functional as F
import common.layers as layers
from common.text.text_processing import TextProcessing
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu


class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None,
                   normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':

        snd, sr = librosa.load(wav)
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel


def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch


def parse_textgrid(tier, sampling_rate, hop_length):

    sil_phones = ["sil", "sp", "spn", ""]
    start_time = tier[0].start_time
    end_time = tier[-1].end_time
    phones = []
    durations = []
    for i, t in enumerate(tier._objects):
        s, e, p = t.start_time, t.end_time, t.text
        if p not in sil_phones:
            phones.append(p)
        else:
            if (i == 0) or (i == len(tier) - 1):
                # leading or trailing silence
                phones.append("sil")
            else:
                # short pause between words
                phones.append("sp")
        durations.append(int(np.ceil(e * sampling_rate / hop_length)
                             - np.ceil(s * sampling_rate / hop_length)))
    n_samples = end_time * sampling_rate
    n_frames = n_samples / hop_length
    # fix occasional length mismatches at the end of utterances when
    # duration in samples is an integer multiple of hop_length
    if n_frames.is_integer():
        durations[-1] += 1

    return phones, durations, start_time, end_time


def extract_durs_from_textgrid(fname, sampling_rate, hop_length, mel_len):

    tg_path = fname
    try:
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
    except FileNotFoundError:
        print('Expected consistent filepaths between wavs and TextGrids, e.g.')
        print('  /path/to/wavs/speaker_uttID.wav -> /path/to/TextGrid/speaker_uttID.TextGrid')
        raise

    phones, durs, start, end = parse_textgrid(
        textgrid.get_tier_by_name('phones'), sampling_rate, hop_length)
    assert sum(durs) == mel_len, f'Length mismatch: {fname}, {sum(durs)} != {mel_len}'

    return torch.LongTensor(durs)


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,
                 dataset_path,
                 audiopaths_and_text,
                 text_cleaners,
                 n_mel_channels,
                 input_type='text',
                 symbol_set='english_basic',
                 p_arpabet=1.0,
                 n_speakers=1,
                 n_conditions=1,
                 load_mel_from_disk=True,
                 load_pitch_from_disk=True,
                 load_duration_from_disk=False,
                 pitch_mean=214.72203,  # LJSpeech defaults
                 pitch_std=65.72038,
                 norm_pitch_by_speaker=False,
                 max_wav_value=None,
                 sampling_rate=None,
                 filter_length=None,
                 hop_length=None,
                 win_length=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 prepend_space_to_text=False,
                 append_space_to_text=False,
                 pitch_online_dir=None,
                 betabinomial_online_dir=None,
                 use_betabinomial_interpolator=True,
                 pitch_online_method='pyin',
                 duration_extraction_method='attn_prior',
                 phrase_level_leg_condition = False,
                 word_level_leg_condition = False,
                 phrase_level_cat_condition = False,
                 word_level_cat_condition = False,
                 **ignored):

        # Expect a list of filenames
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.dataset_path = dataset_path
        # this now returns a list of dicts
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, dataset_path,
            has_speakers=(n_speakers > 1), has_conditions=(n_conditions > 1))

        self.load_mel_from_disk = load_mel_from_disk
        if not load_mel_from_disk:
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)

        self.load_pitch_from_disk = load_pitch_from_disk
        self.load_duration_from_disk = load_duration_from_disk

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text

        assert p_arpabet == 0.0 or p_arpabet == 1.0, (
            'Only 0.0 and 1.0 p_arpabet is currently supported. '
            'Variable probability breaks caching of betabinomial matrices.')

        self.input_type = input_type
        self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)
        self.n_speakers = n_speakers
        self.n_conditions = n_conditions
        self.norm_pitch_by_speaker = norm_pitch_by_speaker
        self.pitch_tmp_dir = pitch_online_dir
        self.f0_method = pitch_online_method
        self.betabinomial_tmp_dir = betabinomial_online_dir
        self.use_betabinomial_interpolator = use_betabinomial_interpolator
        self.duration_extraction_method = duration_extraction_method
        self.hop_length = hop_length
       
        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator()

         
        #-------- word and phrase level conditioning ------#
        # variables a bit verbose, might change
        # currently you have to pass premade tensors so might change too
        self.phrase_level_leg_condition = phrase_level_leg_condition
        self.word_level_leg_condition = word_level_leg_condition
        self.phrase_level_cat_condition = phrase_level_cat_condition
        self.word_level_cat_condition = word_level_cat_condition

        #--------------------------------------------------#

        expected_columns = (2 + int(load_pitch_from_disk) + (n_speakers > 1) + (n_conditions > 1))
        #print('EXPECTED COLUMNS IS ' + str(expected_columns))
        # @TODO Johannah: probably add more asserts for all features
        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)

        if len(self.audiopaths_and_text[0]) < expected_columns:
            raise ValueError(f'Expected {expected_columns} columns in audiopaths file. '
                             'The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>|<condition_id>]')

        if len(self.audiopaths_and_text[0]) > expected_columns:
            print('WARNING: Audiopaths file has more columns than expected')

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x

        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)

    def __getitem__(self, index):
        # Indexing items using dictionary entries
        audiopath = self.audiopaths_and_text[index]['mels']
        #text = self.audiopaths_and_text[index]['text']


        speaker = None
        condition = None
        if self.n_speakers > 1:
            speaker = int(self.audiopaths_and_text[index]['speaker'])
        if self.n_conditions > 1:
            condition = int(self.audiopaths_and_text[index]['condition'])

        #------------------------------------------------------------------
        # for by speaker normalisation (must be found in metadata)

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        if self.norm_pitch_by_speaker:
            pitch_mean = to_tensor(float(self.audiopaths_and_text[index]['pitch_mean']))
            pitch_std = to_tensor(float(self.audiopaths_and_text[index]['pitch_std']))
        else:
            pitch_mean = self.pitch_mean
            pitch_std = self.pitch_std

        #------------------------------------------------------------------

        mel = self.get_mel(audiopath)
   
        #------------------------------------------------------------------
        # if input is phone transcription in metadat, skip preprocessing 
        # and encode directly else, transcribe and encode text

        if self.input_type == 'transcription':
            text = self.get_text(self.audiopaths_and_text[index]['transcription'], encode_transcription=True)
        else:
            text = self.audiopaths_and_text[index]['text']
            text = self.get_text(text)

        pitch = self.get_pitch(index, mel.size(-1), pitch_mean, pitch_std)
        energy = torch.norm(mel.float(), dim=0, p=2)

        #------------------------------------------------------------------
        # duration extraction attn_prior uses built in aligner
        # duration textgrid either parses textgrid or reads tensor or 
        # durations if read from disk is true

        if self.duration_extraction_method == 'attn_prior':
            attn_prior = self.get_prior(index, mel.shape[1], text.shape[0])
            duration = None
        elif self.duration_extraction_method == 'textgrid':
            duration = self.get_durations_textgrid(index, mel.shape[1]) 
            attn_prior = None

        assert pitch.size(-1) == mel.size(-1)

        # No higher formants? -- remove this at some point
        if len(pitch.size()) == 1:
            pitch = pitch[None, :]


        #------------------------------------------------------------------
        # Read in word and phrase conditioning tensors, right now all passed
        # from disk

        phrase_leg = None
        word_leg = None
        phrase_cat = None
        word_cat = None

        if self.phrase_level_leg_condition:
            tensor_path = self.audiopaths_and_text[index]['slope_tensors']
            phrase_leg = torch.load(tensor_path)
            assert phrase_leg.size(dim=1) == len(text)

        if self.word_level_leg_condition:
            tensor_path = self.audiopaths_and_text[index]['leg_word_tensors']
            word_leg = torch.load(tensor_path)
            assert word_leg.size(dim=1) == len(text)

        if self.phrase_level_cat_condition:
            tensor_path = self.audiopaths_and_text[index]['boundary_tensors']
            phrase_cat = torch.load(tensor_path)
            #print(torch.max(phrase_cat), self.audiopaths_and_text[index]['boundary_tensors'])
            assert phrase_cat.size(dim=1) == len(text)

        if self.word_level_cat_condition:
            tensor_path = self.audiopaths_and_text[index]['prominence_tensors']
            word_cat = torch.load(tensor_path)
            assert word_cat.size(dim=1) == len(text)


        return (text, mel, len(text), pitch, energy, speaker, attn_prior,
                audiopath, condition, duration, phrase_leg, word_leg, phrase_cat, word_cat)

    def __len__(self):
        return len(self.audiopaths_and_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm,
                                                 requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            # assert melspec.size(0) == self.stft.n_mel_channels, (
            #     'Mel dimension mismatch: given {}, expected {}'.format(
            #         melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text, encode_transcription=False):
        if encode_transcription:
            text = self.tp.text_to_sequence(text)
        else:
            text = self.tp.encode_text(text)
            space = [self.tp.encode_text("A A")[1]]

            if self.prepend_space_to_text:
                text = space + text

            if self.append_space_to_text:
                text = text + space

        return torch.LongTensor(text)

    def get_prior(self, index, mel_len, text_len):

        if self.use_betabinomial_interpolator:
            return torch.from_numpy(self.betabinomial_interpolator(mel_len,
                                                                   text_len))

        if self.betabinomial_tmp_dir is not None:
            audiopath, *_ = self.audiopaths_and_text[index]
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname = fname.with_suffix('.pt')
            cached_fpath = Path(self.betabinomial_tmp_dir, fname)

            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        attn_prior = beta_binomial_prior_distribution(text_len, mel_len)

        if self.betabinomial_tmp_dir is not None:
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attn_prior, cached_fpath)

        return attn_prior

    def get_durations_textgrid(self, index, mel_len):
        if self.load_duration_from_disk:
            durpath = self.audiopaths_and_text[index]['duration']
            durations = torch.load(durpath)
        else:
            durpath = self.audiopaths_and_text[index]['textgrid']
            durations = extract_durs_from_textgrid(durpath, self.sampling_rate, self.hop_length, mel_len)

        return durations

    def get_pitch(self, index, mel_len=None, pitch_mean=None, pitch_std=None):
        audiopath = self.audiopaths_and_text[index]['mels']

        # why do we need the speaker here?
        spk = 0
        if self.n_speakers > 1:
            spk = int(self.audiopaths_and_text[index]['speaker'])

        if self.load_pitch_from_disk:
            pitchpath = self.audiopaths_and_text[index]['pitch']
            pitch = torch.load(pitchpath)
            if pitch_mean is not None:
                assert pitch_std is not None
                pitch = normalize_pitch(pitch, pitch_mean, pitch_std)
            return pitch

        if self.pitch_tmp_dir is not None:
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname_method = fname.with_suffix('.pt')
            cached_fpath = Path(self.pitch_tmp_dir, fname_method)
            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        # No luck so far - calculate
        wav = audiopath
        if not wav.endswith('.wav'):
            wav = re.sub('/mels/', '/wavs/', wav)
            wav = re.sub('.pt$', '.wav', wav)

        pitch_mel = estimate_pitch(wav, mel_len, self.f0_method,
                                   self.pitch_mean, self.pitch_std)

        if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pitch_mel, cached_fpath)

        return pitch_mel


class TTSCollate:
    """Zero-pads model inputs and targets based on number of frames per step
    Batch indices
    0: text 
    1: mel
    2: len(text)
    3: pitch
    4: energy
    5: speaker
    6: attn_prior
    7: audiopath
    8: condition
    9: duration
    10: phrase_leg
    11: word_leg
    12: phrase_cat
    13: word_cat 
    """

    def __call__(self, batch):
        """Collate training batch from normalized text and mel-spec"""
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # Include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        n_formants = batch[0][3].shape[0]
        pitch_padded = torch.zeros(mel_padded.size(0), n_formants,
                                   mel_padded.size(2), dtype=batch[0][3].dtype)
        energy_padded = torch.zeros_like(pitch_padded[:, 0, :])

        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            energy = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :, :pitch.shape[1]] = pitch
            energy_padded[i, :energy.shape[0]] = energy

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None
      
        if batch[0][6] is not None:
            attn_prior_padded = torch.zeros(len(batch), max_target_len,
                                            max_input_len)
            attn_prior_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                prior = batch[ids_sorted_decreasing[i]][6]
                attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior
        else:
            attn_prior_padded = None

        # Count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        audiopaths = [batch[i][7] for i in ids_sorted_decreasing]

        if batch[0][8] is not None:
            condition = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                condition[i] = batch[ids_sorted_decreasing[i]][8]
        else:
            condition = None

        if batch[0][9] is not None:
            duration_padded = torch.LongTensor(len(batch), max_input_len)
            duration_padded.zero_()
            dur_lens = torch.zeros(duration_padded.size(0), dtype=torch.int32) #added jo 26/10
            for i in range(len(ids_sorted_decreasing)):
                duration = batch[ids_sorted_decreasing[i]][9]
               # duration_padded[i, :duration.size(1)] = duration #changed from 1 for Candor preprocess and LJ preprocess
                duration_padded[i, :duration.shape[0]] = duration #somehow 0 in prepare dataset and 1 in training
                dur_lens[i] = duration.shape[0]

                assert dur_lens[i] == input_lengths[i]
        else:
            duration_padded = None

   # 10: phrase_leg
   # 11: word_leg
   # 12: phrase_cat
   # 13: word_cat

        if batch[0][10] is not None:
            phrase_leg_padded = torch.zeros(len(batch), max_input_len, dtype=batch[0][10].dtype)
            for i in range(len(ids_sorted_decreasing)):
                phrase_leg = batch[ids_sorted_decreasing[i]][10] # [1,seq_len]
                #phrase_leg_padded = F.pad(input=phrase_leg, pad=(0, max_input_len-phrase_leg.size(1)), mode='constant', value=0)
                phrase_leg_padded[i, :phrase_leg.shape[1]] = phrase_leg
        else:
            phrase_leg_padded = None


        if batch[0][11] is not None:
            n_coeffs = batch[0][11].shape[0]
            word_leg_padded = torch.zeros(len(batch), n_coeffs,
                                          max_input_len, dtype=batch[0][11].dtype)
            
            for i in range(len(ids_sorted_decreasing)):
                word_leg = batch[ids_sorted_decreasing[i]][11]
                word_leg_padded[i, :, :word_leg.shape[1]] = word_leg
                #print(word_leg_padded.size())
        else:
            word_leg_padded = None

        if batch[0][12] is not None:
            phrase_cat_padded = torch.zeros(len(batch), max_input_len)
            for i in range(len(ids_sorted_decreasing)):
                phrase_cat = batch[ids_sorted_decreasing[i]][12]
                phrase_cat_padded[i, :phrase_cat.shape[1]] = phrase_cat
        else:
            phrase_cat_padded = None

        if batch[0][13] is not None:
            word_cat_padded = torch.zeros(len(batch), max_input_len)
            for i in range(len(ids_sorted_decreasing)):
                word_cat = batch[ids_sorted_decreasing[i]][13]
                word_cat_padded[i, :word_cat.shape[1]] = word_cat
        else:
            word_cat_padded = None

        return (text_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, speaker, attn_prior_padded,
                audiopaths, condition, duration_padded, phrase_leg_padded,
                word_leg_padded, phrase_cat_padded, word_cat_padded)


def batch_to_gpu(batch):
    (text_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, speaker, attn_prior, audiopaths, 
     condition, duration_padded, phrase_leg_padded, 
     word_leg_padded, phrase_cat_padded, word_cat_padded) = batch

    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    pitch_padded = to_gpu(pitch_padded).float()
    energy_padded = to_gpu(energy_padded).float()

    if attn_prior is not None:
        attn_prior = to_gpu(attn_prior).float()
    if speaker is not None:
        speaker = to_gpu(speaker).long()
    if condition is not None:
        condition = to_gpu(condition).long()
    if duration_padded is not None:
        duration_padded = to_gpu(duration_padded).long()
    if phrase_leg_padded is not None:
        phrase_leg_padded = to_gpu(phrase_leg_padded).float()
    if word_leg_padded is not None:
        word_leg_padded = to_gpu(word_leg_padded).float()
    if phrase_cat_padded is not None:
        phrase_cat_padded = to_gpu(phrase_cat_padded).long()
    if word_cat_padded is not None:
        word_cat_padded = to_gpu(word_cat_padded).long()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, attn_prior, audiopaths, 
         condition, duration_padded, phrase_leg_padded, word_leg_padded,
         phrase_cat_padded, word_cat_padded]
     
    y = [mel_padded, input_lengths, output_lengths, duration_padded]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)

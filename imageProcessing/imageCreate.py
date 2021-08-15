import librosa
import pandas as pd
import numpy as np
import shutil
from skimage.transform import resize
from PIL import Image

from functions import Compose, OneOf, GaussianNoiseSNR, PinkNoiseSNR, PitchShift, TimeShift

def read_data(data_folder):
    data = pd.read_csv(data_folder + '/train_tp.csv')

    return data

def imageCreate(data, data_folder, output_folder, fft, hop, sr, length):

    transform = Compose([
        OneOf([
            GaussianNoiseSNR(),
            PinkNoiseSNR()
        ]),
        PitchShift(),
        TimeShift()
    ])

    for i in range(0, data.shape[0]):
        wav, sr = librosa.load(data_folder + "/train/" + data.iloc[i].recording_id + ".flac", sr = sr)
        t_min = float(data.iloc[i].t_min) * sr
        t_max = float(data.iloc[i].t_max) * sr
        
        # データの切り取り
        center = np.round((t_min + t_max) / 2)
        
        beginning = center - length / 2
        if beginning < 0:
            beginning = 0
            
        ending = beginning + length
        if ending > len(wav):
            ending = len(wav)
            beginning = ending - length
            
        slice = wav[int(beginning):int(ending)]
        slice = transform(slice)
        
        # melspecの準備
        mel_spec = librosa.feature.melspectrogram(
            slice, 
            n_fft = fft, 
            hop_length = hop,
            sr = sr
        )
        mel_spec = librosa.power_to_db(mel_spec)
        mel_spec = resize(mel_spec, (224, 400))
        
        # Normalization
        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)
        
        # 0 ~ 255
        mel_spec = mel_spec * 255
        mel_spec = np.round(mel_spec)
        mel_spec = mel_spec.astype('uint8')
        mel_spec = np.asarray(mel_spec)
        
        img = Image.fromarray(mel_spec)
        img.save(output_folder + data.iloc[i].recording_id + '.png')

    shutil.make_archive(output_folder, 'zip', root_dir=output_folder)
    shutil.rmtree(output_folder)
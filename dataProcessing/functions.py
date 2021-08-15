import numpy  as np
import pandas as pd
import librosa

from PIL import Image
from skimage.transform import resize

def read_data(path):
    train = pd.read_csv(path + '/train_tp.csv')

    return train

def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def get_img(path):
    img = Image.open(path)
    img = np.asarray(img)
    img = mono_to_color(img)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    
    return img

def load_test(test_folder, file_name, params):
    wav, sr = librosa.load(test_folder + "/" + file_name + '.flac', sr = None)
    
    segments = len(wav) / params['length']
    segments = int(np.ceil(segments))
    
    mel_array = []
    
    for i in range(0, segments):
        if (i + 1) * params['length'] > len(wav):
            slice = wav[len(wav) - params['length'] : len(wav)]
        else:
            slice = wav[i * params['length']:(i + 1) * params['length']]
            
        
        mel_spec = librosa.feature.melspectrogram(slice, n_fft = params['fft'], hop_length = params['hop'], sr = params['sr'])
        mel_spec = librosa.power_to_db(mel_spec)
        mel_spec = resize(mel_spec, (224, 400))
        mel_spec = mono_to_color(mel_spec)
        mel_spec = mel_spec / 255.0
        mel_spec = np.transpose(mel_spec, (2, 0, 1))
        
        mel_array.append(mel_spec)
        
    return mel_array
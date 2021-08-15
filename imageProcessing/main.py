import os

from imageCreate import read_data, imageCreate

import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for data processing"
)

parser.add_argument('--data_folder', default='/kaggle/input/rfcx-species-audio-detection', type=str,
                    help='データの入っているフォルダ')
parser.add_argument('--output_folder', default='/kaggle/working/melspec_dataset/', type=str,
                    help="データを出力するフォルダ")
parser.add_argument('--fft', default=2048, type=int)
parser.add_argument('--hop', default=512, type=int)
parser.add_argument('--sr', default=48000, type=int)
parser.add_argument('--length', default=480000, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    os.mkdir(args.output_folder)
    data = read_data(args.data_folder)
    imageCreate(
        data=data,
        data_folder=args.data_folder,
        output_folder=args.output_folder,
        fft=args.fft,
        hop=args.hop,
        sr=args.sr,
        length=args.length
    )
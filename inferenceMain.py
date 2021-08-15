import os

from inference import inference

import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for data processing"
)

parser.add_argument('--data_folder', default='/kaggle/input/rfcx-species-audio-detection', type=str,
                    help='データの入っているフォルダ')
parser.add_argument('--output_folder', default='/kaggle/working', type=str,
                    help="データを出力するフォルダ")
parser.add_argument('--test_folder', default='/kaggle/input/rfcx-species-audio-detection/test', type=str,
                    help="推論時に使うデータが入っているフォルダ")
parser.add_argument('--model_path', default='/kaggle/input/rfcxweight', type=str,
                    help="学習済みモデルが入っているフォルダ")
parser.add_argument('--fft', default=2048, type=int)
parser.add_argument('--hop', default=512, type=int)
parser.add_argument('--sr', default=48000, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    params = {
        'fft': args.fft,
        'hop': args.hop,
        'sr': args.sr,
        'length': args.sr*10
    }
    sub = inference(
       data_folder=args.data_folder,
       test_folder=args.test_folder,
       model_path=args.model_path,
       params=params 
    )
    sub.to_csv(args.output_folder + '/submission.csv', index = False)
from dataProcessing import read_data
from train import train_model
from models import Net
from utils import seed_everything

import warnings
warnings.filterwarnings('ignore')

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for training"
)

parser.add_argument('--data_folder', default='/kaggle/input/rfcx-species-audio-detection', type=str,
                    help='データの入っているフォルダ')
parser.add_argument('--output_folder', default='/kaggle/working', type=str,
                    help="提出用ファイルを出力するフォルダ")
parser.add_argument('--audio_path', default='/kaggle/input/rfcximage', type=str,
                    help='音声をが画像化したデータが入ったフォルダ')
parser.add_argument('--epochs', default=300, type=int,
                    help="学習するエポック数")
parser.add_argument('--es_patience', default=15, type=int,
                    help="何回、改善がないと学習を辞めるか")
args = parser.parse_args()



if __name__ == '__main__':
    seed_everything(71)
    df = read_data(args.data_folder)
    model = Net()
    train_model(
        model=model,
        df=df,
        output_folder=args.output_folder,
        audio_path=args.audio_path,
        epochs=args.epochs,
        es_patience=args.es_patience
    )
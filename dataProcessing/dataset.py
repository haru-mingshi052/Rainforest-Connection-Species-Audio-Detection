from torch.utils.data import Dataset

from .functions import get_img

class RFCxDataset(Dataset):
    def __init__(self, df, audio_path):
        self.df = df
        self.audio_path = audio_path
        
    def __getitem__(self, index):
        x = get_img(self.audio_path + '/' + self.df.iloc[index].recording_id + '.png')
        
        y = self.df.iloc[index].species_id
        
        return x, y
    
    def __len__(self):
        return len(self.df)
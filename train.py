import numpy as np
import time
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataProcessing import RFCxDataset

def train_model(model, df, output_folder, audio_path, epochs, es_patience):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model.to(device)
    
    y = df['species_id']
    
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 47)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(df, y)):
        print('=' * 10, 'Fold', fold, '=' * 10)
        
        train = df.iloc[tr_idx]
        val = df.iloc[val_idx]
    
        criterion = nn.CrossEntropyLoss()
    
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
        scheduler = ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = 'min', 
            patience = 5, 
            verbose = True, 
            factor = 0.2
        )
        
        model_path = output_folder + f'/model_weight_fold{fold}.pth'
        patience = es_patience
        best_train_loss = np.nan
        best_val_loss = np.inf
        best_epoch = np.nan
        
        tr_ds = RFCxDataset(df=train, audio_path=audio_path)
        val_ds = RFCxDataset(df=val, audio_path=audio_path)
        
        tr_dl = DataLoader(dataset = tr_ds, batch_size = 16, shuffle = True)
        val_dl = DataLoader(dataset = val_ds, batch_size = 16, shuffle = False)

        for epoch in range(epochs):
            start_time = time.time()
        
            train_loss = 0.0
            val_loss = 0.0
        
            model.train()
        
            for x, y in tr_dl:
                x = torch.tensor(x, device = device, dtype = torch.float32)
                y = torch.tensor(y, device = device, dtype = torch.long)
                optimizer.zero_grad()
                z = model(x)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(tr_dl)
        
            model.eval()
            with torch.no_grad():
                for x_val, y_val in val_dl:
                    x_val = torch.tensor(x_val, device = device, dtype = torch.float32)
                    y_val = torch.tensor(y_val, device = device, dtype = torch.long)
                    z_val = model(x_val)
                    loss = criterion(z_val, y_val)
                    val_loss += loss.item()
            
                val_loss /= len(val_dl)
            
                finish_time = time.time()
                
                if epoch % 5 == 0:
                    print('Epochs：{:03} | Train Loss：{:.5f} | Val Loss：{:.5f} | Training Time：{:.3f}'
                          .format(epoch, train_loss, val_loss, finish_time - start_time))
            
                scheduler.step(val_loss)
            
                if val_loss <= best_val_loss:
                    best_train_loss = train_loss
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience = es_patience
                
                    torch.save(model, model_path)
                
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping | Epochs：{:03} | Train Loss：{:.5f} | Val Loss：{:.5f}'
                          .format(best_epoch, best_train_loss, best_val_loss))
                        break

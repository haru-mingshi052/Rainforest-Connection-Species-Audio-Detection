import torch
import os
import pandas as pd

from dataProcessing import load_test

def inference(data_folder, test_folder, model_path, params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    sub = pd.read_csv(data_folder + "/sample_submission.csv")
    
    for fold in range(len(os.listdir(model_path))):
        model = torch.load(model_path + f"/model_weight_fold{fold}.pth")
        sub_fold = pd.read_csv(data_folder + '/sample_submission.csv')
    
        model.to(device)
    
        model.eval()
        with torch.no_grad():
            for i in range(0, len(sub)):
                data = load_test(test_folder, sub.recording_id.iloc[i], params)
                data = torch.tensor(data, device = device, dtype  = torch.float32)
            
                output = model(data)
            
                maxed_output = torch.max(output, dim = 0)[0]
                maxed_output = maxed_output.cpu().detach().numpy()
                
                sub_fold.iloc[i,1:] = maxed_output
                
                if fold == 0:
                    sub.iloc[i,1:] = maxed_output
                else:
                    sub.iloc[i,1:] += maxed_output
                    
                if fold == 4:
                    sub.iloc[i,1:] = sub.iloc[i,1:] / 5
                    
        # display(sub_fold.head())
            
    return sub
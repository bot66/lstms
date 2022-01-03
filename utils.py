from torch.utils.data import Dataset
import pandas as pd
import torch

class SpamText(Dataset):
    def __init__(self,csv_path,transform,target_transform={"ham":[1.0,0.0],"spam":[0.0,1.0]}):
        self.df=pd.read_csv(csv_path)
        self.X=self.df["Message"].values
        self.Y=self.df["Category"].values
        self.tokenizer=transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,idx):
        tokens=self.tokenizer(self.X[idx],padding="max_length",max_length=64,truncation=True,return_tensors="pt")
        feat=tokens["input_ids"].squeeze(0)
        attension_mask=tokens["attention_mask"].squeeze(0)
        target=torch.tensor(self.target_transform[self.Y[idx]])
        return  ((feat,attension_mask),target)
    

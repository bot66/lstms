import torch
import torch.nn as nn


class ManyToOne(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,output_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,input_size)
        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.linear=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x=self.embedding(x)
        ouput,(h,c)=self.lstm(x)
        x=self.linear(h.squeeze(0))
        return x 
        
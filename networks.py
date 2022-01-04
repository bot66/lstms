import torch
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.wi=nn.Linear(input_size+hidden_size,hidden_size)
        self.wf=nn.Linear(input_size+hidden_size,hidden_size)
        self.wo=nn.Linear(input_size+hidden_size,hidden_size)
        self.wc=nn.Linear(input_size+hidden_size,hidden_size)
    
    def forward(self,x):
        #input shape [N,L,X]
        h=torch.zeros((x.shape[0],self.hidden_size),dtype=torch.float32) #[N,hidden_size], init hidden state
        c=torch.zeros((x.shape[0],self.hidden_size),dtype=torch.float32) #[N,hidden_size], init cell state
        x=x.permute(1,0,2) #[L,N,X]
        h_states =[] # all timestep hidden states
        for x_i in x:
            inp=torch.cat([x_i,h],dim=1)
            forget_gate=torch.sigmoid(self.wf(inp))
            input_gate=torch.sigmoid(self.wi(inp))
            output_gate=torch.sigmoid(self.wo(inp))
            candidate=torch.tanh(self.wc(inp))
            c=forget_gate.mul(c)+input_gate.mul(candidate)
            h=output_gate.mul(torch.tanh(c))
            h_states.append(h)
        
        return h_states
        

class ManyToOne(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,output_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,input_size)
        self.lstm=LSTM(input_size,hidden_size)

        self.linear=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        #input shape [N,L,X]
        x=self.embedding(x)
        h=self.lstm(x)
        x=self.linear(h[-1])
        return x 

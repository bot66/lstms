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
    
    def forward(self,x,prev_states=None):
        #input shape [N,L,X]
        if not prev_states: # previous states (hidden state,cell state)
            h=torch.zeros((x.shape[0],self.hidden_size),dtype=torch.float32).to(x.device) #[N,hidden_size], init hidden state
            c=torch.zeros((x.shape[0],self.hidden_size),dtype=torch.float32).to(x.device) #[N,hidden_size], init cell state
        else:
            h,c=prev_states
        x=x.permute(1,0,2) #[L,N,X]
        h_states =[] # all timestep hidden states
        c_states = [] # all timestep cell states
        for x_i in x:
            inp=torch.cat([x_i,h],dim=1)
            forget_gate=torch.sigmoid(self.wf(inp))
            input_gate=torch.sigmoid(self.wi(inp))
            output_gate=torch.sigmoid(self.wo(inp))
            candidate=torch.tanh(self.wc(inp))
            c=forget_gate.mul(c)+input_gate.mul(candidate)
            h=output_gate.mul(torch.tanh(c))
            h_states.append(h)
            c_states.append(c)

        return (h_states[-1],h_states,c_states) # last hidden state, hidden states, cell states

class Attention(nn.Module):
#attention in seq2seq,reproduce "Luong attention"(https://arxiv.org/pdf/1508.04025.pdf)
    def __init__(self,decoder_hidden_size,encoder_hidden_size):
        super().__init__()
        self.linear=nn.Linear(decoder_hidden_size,encoder_hidden_size)
    def forward(self,x,encoder_states):
        x=self.linear(x)
        x=x.matmul(encoder_states)

        return x
        
        
# many to one
class ManyToOne(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,output_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,input_size,padding_idx=0)
        self.lstm=LSTM(input_size,hidden_size)
        self.linear=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        #input shape [N,L,X]
        x=self.embedding(x)
        h,h_s,c_s=self.lstm(x)
        x=self.linear(h)

        return x 

#many to many
class Encoder(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,input_size,padding_idx=0)
        self.lstm=LSTM(input_size,hidden_size)

    def forward(self,x):
        #input shape [N,L,X]
        x=self.embedding(x)
        h_states=self.lstm(x)

        return (h_states[-1],h_states) 

class Decoder(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,input_size,padding_idx=0)
        self.lstm=LSTM(input_size,hidden_size)
        self.linear=nn.Linear(hidden_size,vocab_size)
    def forward(self,x,prev_states):
        #input shape [N,L,X]
        x=self.embedding(x)
        h=self.lstm(x)
        x=self.linear(h[-1])

        return x
 
class ManyToMany(nn.Module):
    #seq2seq with "Luong attention"(https://arxiv.org/pdf/1508.04025.pdf)
    def __init__(self,\
                enc_vocab_size,\
                dec_vocab_size,\
                enc_emb_size,\
                dec_emb_size,\
                hidden_size):
        super().__init__()
        self.ouput_size=dec_vocab_size
        self.encoder=Encoder(enc_vocab_size,enc_emb_size,hidden_size)
        self.decoder=Decoder(dec_vocab_size,dec_emb_size,hidden_size)
        self.attention=None
        
    def forward(self,x,y):
        #input shape [N,L,X]
        context,enc_hidden_states=self.encoder(x)
        #[BOS]
        bos=x[:,0,:]

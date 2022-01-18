import torch
import torch.nn as nn
import random

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
    def forward(self,dec_h,encoder_states):
        x=self.linear(dec_h).unsqueeze(1)
        stk_enc_states=torch.stack(encoder_states,dim=2)
        score=torch.bmm(x,stk_enc_states)
        attn_weight=torch.softmax(score,dim=2)

        context=torch.bmm(attn_weight,stk_enc_states.permute(0,2,1))
        #combine dec_h and context
        combined_h=torch.cat([dec_h,context.squeeze(1)],dim=1) 

        return combined_h
        
        
# many to one
class ManyToOne(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size,output_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,input_size,padding_idx=0)
        self.lstm=LSTM(input_size,hidden_size)
        self.linear=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        #input shape [N,L]
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
        #input shape [N,L]
        x=self.embedding(x)
        h,h_states,c_states=self.lstm(x)

        return (h,h_states,c_states) 

class Decoder(nn.Module):
    def __init__(self,vocab_size,input_size,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,input_size,padding_idx=0)
        self.lstm=LSTM(input_size,hidden_size)
    def forward(self,x,prev_states):
        #input shape [N,L]
        x=self.embedding(x)
        h,h_states,c_states=self.lstm(x,prev_states)

        return (h,h_states,c_states) 
 
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
        self.attention=Attention(hidden_size,hidden_size)
        self.linear=nn.Linear(2*hidden_size,hidden_size)
        self.out=nn.Linear(hidden_size,dec_vocab_size)
        
    def forward(self,x,y):
        #input shape [N,L]
        enc_h,enc_hidden_states,enc_cell_states=self.encoder(x)
        prev_h=enc_h # previous hidden state in decode step
        prev_c=enc_cell_states[-1] # previous cell state in decode step
        #[BOS]
        inp=y[:,0:1]
        pred_logits=[]
        for i in range(1,y.shape[1]):
            dec_h,dec_hidden_states,dec_cell_states=self.decoder(inp,\
                                                                (prev_h,prev_c))
            combined_h = self.attention(dec_h,enc_hidden_states)
            logits=self.linear(combined_h)
            logits=torch.tanh(logits)
            logits=self.out(logits)
            pred_logits.append(logits)
            pred_id=logits.softmax(dim=1).argmax(dim=1,keepdim=True)
            #teacher forcing at thresh 0.5
            inp= y[:,i:i+1] if random.random()>0.5 else pred_id
            #reset previous step states
            prev_h=dec_h
            prev_c=dec_cell_states[-1]

        return torch.stack(pred_logits,dim=1)

    @torch.no_grad()
    def inference(self,x,eos_id=2,max_words=20):
        #simple greedy search
        #input shape [N,L]
        enc_h,enc_hidden_states,enc_cell_states=self.encoder(x)
        prev_h=enc_h # previous hidden state in decode step
        prev_c=enc_cell_states[-1] # previous cell state in decode step
        #[BOS]
        inp=x[:,0:1]
        pred_ids=[]
        for i in range(max_words):
            dec_h,dec_hidden_states,dec_cell_states=self.decoder(inp,\
                                                                (prev_h,prev_c))
            combined_h = self.attention(dec_h,enc_hidden_states)
            logits=self.linear(combined_h)
            logits=torch.tanh(logits)
            logits=self.out(logits)
            pred_id=logits.softmax(dim=1).argmax(dim=1,keepdim=True)
            pred_ids.append(pred_id)
            #teacher forcing at thresh 0.5
            inp=pred_id
            #reset previous step states
            prev_h=dec_h
            prev_c=dec_cell_states[-1]

        return torch.cat(pred_ids,dim=1)


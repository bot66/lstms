from tokenizers import Tokenizer
import torch
from utils import EnglishToChinese
from networks import ManyToMany
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

def run():
    #English tokenizer
    enc_tokenizer = Tokenizer.from_file("datasets/machine_translation/machine-translation-en.json")
    #Chinese tokenizer
    dec_tokenizer = Tokenizer.from_file("datasets/machine_translation/machine-translation-cn.json")

    model=ManyToMany(enc_vocab_size=enc_tokenizer.get_vocab_size(),\
                    dec_vocab_size=dec_tokenizer.get_vocab_size(),\
                    enc_emb_size=1000,\
                    dec_emb_size=2000,\
                    hidden_size=500)
    #train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #hyp
    epoches = 1
    lr=0.01
    batch_size=8

    train_dataloader=DataLoader(EnglishToChinese("datasets/machine_translation/cmn_train.txt",enc_tokenizer,dec_tokenizer),\
                                        batch_size,
                                        shuffle=True)
    optimizer=AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    criteria=nn.CrossEntropyLoss(ignore_index=0) # ignore padding id
    schedule=MultiStepLR(optimizer,milestones=[int(epoches*0.6),int(epoches*0.8)], gamma=0.1) 

    for e in range(epoches):
        running_loss = 0.0
        for i,b in enumerate(train_dataloader):
            optimizer.zero_grad()
            logits=model(b[0].to(device),b[1].to(device))
            break
            # loss = criteria(logits,b[1].to(device))
            # loss.backward()
            # optimizer.step()            

if __name__=="__main__":
    run()
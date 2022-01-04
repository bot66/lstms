import torch
from transformers import BertTokenizerFast
from utils import SpamText
from networks import ManyToOne
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

def run():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model= ManyToOne(vocab_size=tokenizer.vocab_size,\
                    input_size=1000,\
                    hidden_size=128,\
                    output_size=2)
    
    #train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #hyp
    epoches = 30
    lr=0.01
    batch_size=128

    train_dataloader=DataLoader(SpamText("datasets/text_classification/SPAM-text-message_20170820_Data_train.csv",tokenizer),\
                                        batch_size,
                                        shuffle=True)
    optimizer=AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    criteria=nn.CrossEntropyLoss()
    schedule=MultiStepLR(optimizer,milestones=[int(epoches*0.6),int(epoches*0.8)], gamma=0.1)
    for e in range(epoches):
        running_loss = 0.0
        for i,b in enumerate(train_dataloader):
            optimizer.zero_grad()
            logits=model(b[0][0].to(device))
            loss = criteria(logits,b[1].to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9: # print average loss every 10 mini batches
                print('[Epoch:%d, %5d / %5d] loss: %.6f lr: %.6f' %
                    (e + 1, i + 1, len(train_dataloader),running_loss / 10,schedule.get_last_lr()[0]))
                running_loss = 0.0
        schedule.step()
    #test
    test_dataloader=DataLoader(SpamText("datasets/text_classification/SPAM-text-message_20170820_Data_test.csv",tokenizer),\
                                        batch_size)
    with torch.no_grad():
        model.eval()
        total=0
        correct=0
        for i,b in enumerate(test_dataloader):
            logits=model(b[0][0].to(device))
            pred=logits.softmax(dim=1).argmax(dim=1).data
            target=b[1].to(device).argmax(dim=1).data
            _=(pred==target)
            correct+=_.sum().item()
            total+=len(_)
            
        print("test acc:",correct/total)

if __name__ == "__main__":
    run()
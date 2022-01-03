import torch
from transformers import BertTokenizerFast
from utils import SpamText,create_dataloader
from networks import ManyToOne
from torch.optim import Adam
from torch.utils.data import DataLoader

def train():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataloader=DataLoader(SpamText("datasets/text_classification/SPAM-text-message_20170820_Data_train.csv",tokenizer),\
                                        32)

    model= ManyToOne(vocab_size=tokenizer.vocab_size,\
                    input_size=1000,\
                    hidden_size=128,\
                    output_size=2)
    

def test():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    test_dataloader=DataLoader(SpamText("datasets/text_classification/SPAM-text-message_20170820_Data_test.csv",tokenizer),\
                                        4)
    model= ManyToOne(vocab_size=tokenizer.vocab_size,\
                    input_size=1000,\
                    hidden_size=128,\
                    output_size=2)

if __name__ == "__main__":
    train()
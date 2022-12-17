import os
import sys
import argparse
from tqdm import trange
# from torchtext import data as torchtext_data
# from torchtext import datasets

import torch
import torch.utils.data as data

# from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import torch.optim
import torch.nn.functional as F
import numpy as np
from IPython import embed
from operator import add
# from run_gpt2 import top_k_logits
# from style_utils import to_var
import copy
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim 

import jittor as jt

from jittor.dataset import Dataset as jtDataset

from torchtext import data as torchtext_data
from torchtext import datasets

from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import transformers
import random
#=====================================Dataset & DataLoader=======================
class myDataset(jt.dataset.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = jt.array(y)
        lengths = [len(seq) for seq in x]
        padding = jt.zeros(len(lengths), max(lengths))
        for i, seq in enumerate(x):
            padding[i,:lengths[i]] = seq[:lengths[i]]
        self.x = padding
    
    def __getitem__(self, k):
        return self.x[k], self.y[k]
    
    def __len__(self):
        return len(self.y)
# class myDataLoader(jt.dataset.Dataset):
#     def __init__(self, dataset, batch_size = 64, shuffle = False):
#         super().__init__()
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.x = []
#         self.y = []
#         for i in range(len(dataset) // batch_size):
#             x, y = dataset[i*batch_size:(i+1)*batch_size]
#             self.x.append(x)
#             self.y.append(y)
#         self.map = [i for i in range(len(self.x))]
#         self.map = random.shuffle(self.map)
#     def __getitem__(self, k):
#         if self.shuffle:
#             return self.x[self.map[k]], self.y[self.map[k]]
#         else:
#             return self.x[k], self.y[k]
        
#     def __len__(self):
#         return len(self.x)
        
def myDataLoader(data: myDataset, batch_size=64):
    for i in range(len(data)//batch_size):
        yield data[batch_size*i : batch_size*i+batch_size]

#================================================================================

#=======================ClassificationHead & Discriminator=======================
class myClassificationHead(jt.nn.Module):
    def __init__(self, class_size=5, embed_size=2048):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = jt.nn.Linear(embed_size, class_size)
    def execute(self, hidden_state):
        lm_logits = self.mlp(hidden_state)
        return lm_logits

class myDiscriminator2mean(jt.nn.Module):
    def __init__(self, class_size=5, embed_size=1024, head=None):
        super().__init__()
        if head == None:
            self.classifierhead = myClassificationHead(class_size=class_size, embed_size=embed_size)
        else:
            self.classifierhead = head
        self.model = model
        self.embed_size = embed_size
    
    def get_classifier(self):
        return self.classifierhead

    def train_custom(self):
        for param in self.model.parameters():
            param.requires_grad = False
        pass
        self.classifierhead.train()

    def execute(self, x):
        mask_src = 1 - x.equal(0).unsqueeze(1).detach()
        mask_src = mask_src.repeat(1, self.embed_size, 1) #batch_size, 1024, length (repeat each sentence for 1024 times)

        x = x.tolist()
        x = jt.array(x,dtype=torch.long)
        output_dict = model(x, output_hidden_states=True)
        hidden = output_dict.hidden_states[-1]
        # x = model.forward_embed(x)
        # hidden, x = model.forward_transformer_embed(x)
        #  Hidden has shape batch_size x length x embed-dim
        hidden = hidden.tolist()
        hidden = jt.array(hidden)

        hidden = hidden.permute(0, 2, 1)
        _, _, batch_length = hidden.shape
        hidden = hidden * mask_src  # / torch.sum(mask_src, dim=-1).unsqueeze(2).repeat(1, 1, batch_length)
        #
        hidden = hidden.permute(0, 2, 1)
        x =  jt.sum(hidden, dim=1)/(jt.sum(mask_src, dim=-1).detach() + 1e-10)
        x = self.classifierhead(x)
        x = jt.nn.log_softmax(x, dim=-1)
        return x
#================================================================================


#======================================Train & test==================================
def train_epoch(data_loader, discriminator, device='cpu', args=None, epoch=1):
    optimizer = jt.optim.Adam(discriminator.parameters(), lr=0.0001)
    # optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator.train_custom()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data, target # data is 2-d list [batch_size, length(after padding)], target is 1-d list [batch_size]

        optimizer.zero_grad()

        output = discriminator(data)
        loss = jt.nn.nll_loss(output, target)
        optimizer.backward(loss)
        #loss.backward(retain_graph=True)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Relu Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 318582,
                       batch_idx * len(data) / 318582, loss.item()))

def test_epoch(data_loader, discriminator, device='cpu', args=None):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with jt.no_grad():
        for data, target in data_loader:
            #ata, target = data.to(device), target.to(device)
            output = discriminator(data)
            test_loss += jt.nn.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred,_ = output.argmax(dim=1, keepdims=True)  # get the index of the max log-probability
            correct += pred.equal(target.reshape(pred.shape)).sum().item()
            #print(pred, target)

    test_loss /= 2210

    print('\nRelu Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, 2210,
        100. * correct / 2210))
#================================================================================
jt.flags.use_cuda = jt.has_cuda

parser = argparse.ArgumentParser(description='Train a discriminator on top of GPT-2 representations')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='Number of training epochs')
parser.add_argument('--save-model', action='store_true', help='whether to save the model')
parser.add_argument('--dataset-label', type=str, default='SST',choices=('SST', 'clickbait', 'toxic'))
args = parser.parse_args()
batch_size = args.batch_size

tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium')

text = torchtext_data.Field()
label = torchtext_data.Field(sequential=False)
train_data, val_data, test_data = datasets.SST.splits(text, label, fine_grained=True, train_subtrees=True,
                                                        # filter_pred=lambda ex: ex.label != 'neutral'
                                                        )

x = []
y = []
d = {"positive": 0, "negative": 1, "very positive": 2, "very negative": 3, "neutral": 4}

for i in range(len(train_data)):
    seq = TreebankWordDetokenizer().detokenize(vars(train_data[i])["text"])
    seq = tokenizer.encode(seq)
    # seq = jt.array(seq)
    x.append(seq)
    y.append(d[vars(train_data[i])["label"]])
jt_dataset = myDataset(x,y)
jt_DataLoader = myDataLoader(jt_dataset, batch_size=args.batch_size)

test_x = []
test_y = []
for i in range(len(test_data)):
    seq = TreebankWordDetokenizer().detokenize(vars(test_data[i])["text"])
    seq = tokenizer.encode(seq)
    seq = [50256] + seq
    #seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
    test_x.append(seq)
    test_y.append(d[vars(test_data[i])["label"]])
jt_test_dataset = myDataset(test_x, test_y)
jt_test_dataloader = myDataLoader(jt_test_dataset)

test_head = myClassificationHead(class_size=5,embed_size=1024)
test_head.load_state_dict(torch.load('discrim_models/sentiment_classifierhead.pt',map_location=torch.device('cpu')))
test_dis = myDiscriminator2mean(head=test_head)

test_epoch(jt_test_dataloader, test_dis)
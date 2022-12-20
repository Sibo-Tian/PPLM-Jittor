# %%
import argparse
from tqdm import trange
# from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext import data as torchtext_data
from torchtext import datasets

import jittor as jt
import gpt2
import time

import transformers

# %%
class myDataset(jt.dataset.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x #2d
        self.y = jt.array(y) #1d
        lengths = [len(seq) for seq in x]
        padding = jt.zeros(len(lengths), max(lengths))
        for i, seq in enumerate(x):
            padding[i,:lengths[i]] = seq[:lengths[i]]
        self.x = padding
    
    def __getitem__(self, k):
        return self.x[k], self.y[k]
    
    def __len__(self):
        return len(self.y)
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
# model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium')

# %%
class ClassificationHead(jt.nn.Module):
    def __init__(self, class_size=5, embed_size=2048):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.mlp = jt.nn.Linear(embed_size, class_size)
    def execute(self, hidden_state):
        lm_logits = self.mlp(hidden_state)
        return lm_logits

class Discriminator2mean(jt.nn.Module):
    def __init__(self, class_size=5, embed_size=1024, head=None):
        super().__init__()
        if head == None:
            self.classifierhead = ClassificationHead(class_size=class_size, embed_size=embed_size)
        else:
            self.classifierhead = head
        config = gpt2.GPT2Config()
        self.model = gpt2.GPT2LMHeadModel(config)
        self.model.load('gpt2.pkl')
        self.embed_size = embed_size
    
    def get_classifier(self):
        return self.classifierhead

    def get_classifier_param(self):
        return self.classifierhead.parameters()

    def execute(self, x):
        mask_src = 1 - x.equal(0).unsqueeze(1).detach()
        mask_src = mask_src.repeat(1, self.embed_size, 1) #batch_size, 1024, length (repeat each sentence for 1024 times)

        output_dict = self.model(x)
        hidden = output_dict.hidden_states[-1]

        hidden = hidden.permute(0, 2, 1)
        hidden = hidden * mask_src  # / torch.sum(mask_src, dim=-1).unsqueeze(2).repeat(1, 1, batch_length)
        #
        hidden = hidden.permute(0, 2, 1)
        x =  jt.sum(hidden, dim=1)/(jt.sum(mask_src, dim=-1) + 1e-10)
        x = self.classifierhead(x)
        x = jt.nn.log_softmax(x, dim=-1)
        return x

# %%
jt.flags.use_cuda = jt.has_cuda

parser = argparse.ArgumentParser(description='Train a discriminator on top of GPT-2 representations')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='Number of training epochs')
parser.add_argument('--save_path', type=str, default='', help='whether to save the model')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--dataset_label', type=str, default='SST',choices=('SST', 'clickbait', 'toxic'))
args = parser.parse_args(args=[])

# %%
if args.dataset_label == 'SST':
    text = torchtext_data.Field()
    label = torchtext_data.Field(sequential=False)
    train_data, val_data, test_data = datasets.SST.splits(text, label, fine_grained=True, train_subtrees=True,
                                                            # filter_pred=lambda ex: ex.label != 'neutral'
                                                            )
    d = {"positive": 0, "negative": 1, "very positive": 2, "very negative": 3, "neutral": 4}
    x = []
    y = []
    for i in range(len(train_data)):
        seq = TreebankWordDetokenizer().detokenize(vars(train_data[i])["text"])
        seq = tokenizer.encode(seq)
        x.append(seq)
        y.append(d[vars(train_data[i])["label"]])
    train_dataset = myDataset(x,y).set_attrs(batch_size=args.batch_size,shuffle=True)

    test_x = []
    test_y = []
    for i in range(len(test_data)):
        seq = TreebankWordDetokenizer().detokenize(vars(test_data[i])["text"])
        seq = tokenizer.encode(seq)
        seq = [50256] + seq
        #seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
        test_x.append(seq)
        test_y.append(d[vars(test_data[i])["label"]])
    test_dataset = myDataset(test_x, test_y).set_attrs(batch_size=args.batch_size,shuffle=True)
elif args.dataset_label == 'clickbait':
    pass
else:
    pass

# %%
def train_epoch(data_loader, discriminator:Discriminator2mean, args=None):
    optimizer = jt.optim.Adam(discriminator.get_classifier_param(), lr=0.0001)
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            print('Epoch: {}, batch: {}'.format(idx,batch_idx))
            start = time.time()
            data, target = data, target.reshape(-1) # data is 2-d list [batch_size, length(after padding)], target is 1-d list [batch_size]
            optimizer.zero_grad()
            output = discriminator(data)
            loss = jt.nn.nll_loss(output, target)
            optimizer.step(loss)
            print('batch time cost: {}'.format(time.time() - start))
            if batch_idx % args.log_interval == 0:
                print('Relu Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader),
                        batch_idx * len(data) / len(data_loader), loss.item()))
        head = discriminator.get_classifier()
        head.save(args.dataset_label+'-'+str(epoch)+'.pkl')

# %%
def test_epoch(data_loader, discriminator, args=None):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with jt.no_grad():
        for data, target in data_loader:
            output = discriminator(data)
            test_loss += jt.nn.nll_loss(output, target.reshape(-1)).item()  # sum up batch loss
            pred,_ = output.argmax(dim=1, keepdims=True)  # get the index of the max log-probability
            correct += pred.equal(target.reshape(pred.shape)).sum().item()

    print('\nRelu Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader),
        100. * correct / len(data_loader)))

# %%
model = Discriminator2mean()
train_epoch(train_dataset, model, args)



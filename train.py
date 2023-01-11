# %%
import argparse
from tqdm import trange
import datasets
import jittor as jt
import gpt2
import time
import transformers
from classification import ClassificationHead, Discriminator2mean
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
config = gpt2.GPT2Config()
model = gpt2.GPT2LMHeadModel(config)
model.load('gpt2-medium.pkl')

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

class cacheDataset(jt.dataset.Dataset):
    def __init__(self, x, y, model):
        super().__init__()
        self.x = x
        self.y = jt.array(y) #1d
        lengths = [len(seq) for seq in x]
        padding = jt.zeros(len(lengths), max(lengths))
        for i, seq in enumerate(x):
            padding[i,:lengths[i]] = seq[:lengths[i]]
        self.x = padding
        #cache
        mask_src = 1 - x.equal(0).unsqueeze(1).detach()
        mask_src = mask_src.repeat(1, self.embed_size, 1)

        result = model(self.x)
        hidden = result['hidden_states'][-1]

        hidden = hidden.permute(0, 2, 1)
        hidden = hidden * mask_src  
        hidden = hidden.permute(0, 2, 1)

        self.x =  jt.sum(hidden, dim=1)/(jt.sum(mask_src, dim=-1) + 1e-10)
    def __getitem__(self, k):
        return self.x[k], self.y[k]
    
    def __len__(self):
        return len(self.y)

# %%
def train_epoch(data_loader, discriminator:Discriminator2mean, args=None):
    optimizer = jt.optim.Adam(discriminator.get_classifier_param(), lr=args.lr)
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            print('Epoch: {}, batch: {}'.format(epoch,batch_idx))
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

        
def train_cache(data_loader, classificationHead, args):
    optimizer = jt.optim.Adam(classificationHead.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            print('Epoch: {}, batch: {}'.format(epoch,batch_idx))
            start = time.time()
            data, target = data, target.reshape(-1) # data is 2-d list [batch_size, length(after padding)], target is 1-d list [batch_size]
            optimizer.zero_grad()
            output = classificationHead(data)
            loss = jt.nn.nll_loss(output, target)
            optimizer.step(loss)
            print('batch time cost: {}'.format(time.time() - start))
            if batch_idx % args.log_interval == 0:
                print('Relu Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(data_loader),
                        batch_idx * len(data) / len(data_loader), loss.item()))
        classificationHead.save(args.dataset_label+'-'+str(epoch)+'.pkl')

    
def test_epoch(data_loader, discriminator, args=None):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with jt.no_grad():
        idx = 1
        for data, target in data_loader:
            print('batch:{} / total{}'.format(idx, len(data_loader)))
            idx += 1
            output = discriminator(data)
            test_loss += jt.nn.nll_loss(output, target.reshape(-1)).item()  # sum up batch loss
            pred,_ = output.argmax(dim=1, keepdims=True)  # get the index of the max log-probability
            correct += pred.equal(target.reshape(pred.shape)).sum().item()

    print('\nRelu Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader),
        100. * correct / len(data_loader)))
    positive_example = 'I love you'
    negative_example = 'I hate you'
    positive_encoded = jt.array(tokenizer.encode(positive_example)).unsqueeze(dim=0)
    negative_encoded = jt.array(tokenizer.encode(negative_example)).unsqueeze(dim=0)
    positive_res = discriminator(positive_encoded).squeeze(dim=0)
    negative_res = discriminator(negative_encoded).squeeze(dim=0)
    print(positive_example,' :','"positive":{}, "negative": {}, "very positive": {}, "very negative": {}, "neutral": {} \n'.format(positive_res[0],
                                                                                positive_res[1],positive_res[2],positive_res[3],positive_res[4]))
    print(negative_example,' :','"positive":{}, "negative": {}, "very positive": {}, "very negative": {}, "neutral": {} \n'.format(negative_res[0],
                                                                                negative_res[1],negative_res[2],negative_res[3],negative_res[4]))


# %%
jt.flags.use_cuda = jt.has_cuda
parser = argparse.ArgumentParser(description='Train a discriminator on top of GPT-2 representations')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='Number of training epochs')
parser.add_argument('--lr',type=float, default=0.0001)
parser.add_argument('--save_path', type=str, default='', help='whether to save the model')
parser.add_argument('--dataset_label', type=str, default='SST')
parser.add_argument('--cache_hidden',action='store_true')
args = parser.parse_args(args=['--cache_hidden'])

# %%
if args.dataset_label == 'SST':
    raw_train_dataset = datasets.load_dataset('sst2', split='train')
    raw_test_dataset = datasets.load_dataset('sst2', split='validation')#use validation dataset, as no labels for test dataset
    x = [[50256]+tokenizer.encode(sen) for sen in raw_train_dataset['sentence']]
    y = raw_train_dataset['label']
    if not args.cache_hidden:
        train_dataset = myDataset(x,y).set_attrs(batch_size=args.batch_size,shuffle=True)
    else:
        train_dataset = cacheDataset(x,y,model).set_attrs(batch_size=args.batch_size,shuffle=True)
    x = [[50256]+tokenizer.encode(sen) for sen in raw_test_dataset['sentence']]
    y = raw_test_dataset['label']
    if not args.cache_hidden:
        test_dataset = myDataset(x,y).set_attrs(batch_size=args.batch_size,shuffle=True)
    else:
        test_dataset = cacheDataset(x,y,model).set_attrs(batch_size=args.batch_size,shuffle=True)
else:
    raise Exception('Not support for this dataset...')

# %%
model = Discriminator2mean()

# %%
train_epoch(train_dataset, model, args)



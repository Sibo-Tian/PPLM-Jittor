# %%
import argparse
from tqdm import trange
from operator import add

import torch
import transformers

import jittor as jt
from jittor import nn
import gpt2

from style_utils import top_k_logits
from scores import dist_n

# %%
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
    def __init__(self, class_size=5, embed_size=1024, head=None, model=None):
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
        output_dict = self.model(x, output_hidden_states=True)
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

# %%
SmallConst = 1e-15
copy_model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium')
enc = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
config = gpt2.GPT2Config()
m = gpt2.GPT2Model(config)
m.load_state_dict(copy_model.transformer.state_dict())
l = jt.nn.Linear(config.n_embd, config.vocab_size, bias=False)
l.load_state_dict(copy_model.lm_head.state_dict())
model = gpt2.GPT2LMHeadModel(config, m, l)

def latent_perturb(model, args, context=None, sample=True):
    #==================================================prepare the discriminator/bag of words==============================================
    if args.discrim == 'clickbait':
        classifier = myClassificationHead(class_size=2, embed_size=1024)
        classifier.load_state_dict(torch.load("discrim_models/clickbait_classifierhead.pt"))
        classifier.eval()
        args.label_class = 1 # clickbaity
    #TODO map_location=torch.device('cpu')
    elif args.discrim == 'sentiment':
        classifier = myClassificationHead(class_size=5, embed_size=1024)
        classifier.load_state_dict(torch.load("discrim_models/sentiment_classifierhead.pt",map_location=torch.device('cpu')))
        classifier.eval()
        if args.label_class < 0:
            raise Exception('Wrong class for sentiment, use --label-class 2 for *very positive*, 3 for *very negative*')
        #args.label_class = 2 # very pos
        #args.label_class = 3 # very neg

    elif args.discrim == 'toxicity':
        classifier = myClassificationHead(class_size=2, embed_size=1024)
        classifier.load_state_dict(torch.load("discrim_models/toxicity_classifierhead.pt"))
        classifier.eval()
        args.label_class = 0 # not toxic
    else:
        classifier = None
    
    # Get tokens for the list of positive words
    def list_tokens(word_list):
        token_list = []
        for word in word_list:
            token_list.append(enc.encode(" " + word))
        return token_list

    good_index = []
    if args.bag_of_words:
        bags_of_words = args.bag_of_words.split(";")
        for wordlist in bags_of_words:
            with open(wordlist, "r") as f:
                words = f.read()
                words = words.split('\n')
            good_index.append(list_tokens(words)) # good_index is the encode of the words
        
    if args.bag_of_words and classifier:
        print('Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.')
        args.loss_type = 3

    elif args.bag_of_words:
        args.loss_type = 1
        print('Using PPLM-BoW')

    elif classifier is not None:
        args.loss_type = 2
        print('Using PPLM-Discrim')

    else:
        raise Exception('Supply either --bag-of-words (-B) or --discrim -D')
    
    #==================================================generate non perturbed words=======================================================
    original, _, _ = sample_from_hidden(model=model, args=args, context=context,
                                  perturb=False, good_index=good_index, classifier=classifier)

    #==================================================generate perturbed words============================================================
    perturbed_list = []
    discrim_loss_list = []
    loss_in_time_list = []

    for i in range(args.num_samples): #num_samples : how many output words
        perturbed, discrim_loss, loss_in_time = sample_from_hidden(model=model, args=args, context=context,
                                                        perturb=True, good_index=good_index,
                                                         classifier=classifier)
        perturbed_list.append(perturbed)
        if classifier is not None:
            discrim_loss_list.append(discrim_loss)
        loss_in_time_list.append(loss_in_time)
    
    return original, perturbed_list, discrim_loss_list, loss_in_time_list


# %%
def sample_from_hidden(model, args, classifier, context=None, past=None,
                       sample=False, perturb=True, good_index=None):
    
    output = jt.int64(context).unsqueeze(0) if context else None
    #output: 2-d list, with jt.int64 element, the format required by gpt (input_id)
    grad_norms = None
    loss_in_time = []
    #iteration for args.length times, the output sentence's length = original + args.length
    for i in trange(args.length, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        # Therefore, use everything from before current i/p token to generate relevant past

        if past is None and output is not None:
            #=======================devide the output(context) into 2 parts : past and one word for query=======================
            prev = output[:, -1:]
            res =  model(output[:, :-1])
            past = res['past_key_values']

            res = model(output)
            original_probs, true_past = res['logits'], res['past_key_values']
            true_hidden = res['hidden_states'][-1]

        else:
            res = model(output)
            original_probs, true_past = res['logits'], res['past_key_values']
            true_hidden = res['hidden_states'][-1]

        #TODO
        if i >= args.grad_length:
            current_stepsize = args.stepsize * 0
        else:
            current_stepsize = args.stepsize

        #not perturb
        if not perturb or args.num_iterations == 0:
            perturbed_past = past

        #perturb
        else:
        #==============================================perturb the past==============================================
            accumulated_hidden = true_hidden[:, :-1, :] #all hidden states except query word
            accumulated_hidden = jt.sum(accumulated_hidden, dim=1)
            perturbed_past, _, grad_norms, loss_per_iter = perturb_past(past, model, prev, args,
                                                                        classifier=classifier,
                                                                        good_index=good_index, stepsize=current_stepsize,
                                                                        original_probs=original_probs,
                                                                        true_past=true_past,
                                                                        accumulated_hidden=accumulated_hidden,
                                                                        grad_norms=grad_norms)
            loss_in_time.append(loss_per_iter)
        #==============================================use the query word to 'query' past=============================
        res = model(prev, past_key_values=perturbed_past)
        test_logits, past, hidden = res['logits'], res['past_key_values'], res['hidden_states'][-1]
         
        # test_logits = F.softmax(test_logits[:, -1, :], dim=-1)
        # likelywords = torch.topk(test_logits, k=10, dim=-1)
        # print(enc.decode(likelywords[1].tolist()[0]))

        if classifier is not None:
            ce_loss = jt.nn.CrossEntropyLoss()
            predicted_sentiment = classifier(jt.mean(true_hidden, dim=1))
            label = jt.int64([args.label_class])
            true_discrim_loss = ce_loss(predicted_sentiment, label)
            print("true discrim loss", true_discrim_loss)
        else:
            true_discrim_loss = 0 

        
        logits = test_logits
        logits = logits[:, -1, :] / args.temperature  # + SmallConst

        log_probs = jt.nn.softmax(logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:
            original_probs = jt.nn.softmax(original_probs[:, -1, :], dim=-1)
            gm_scale = args.fusion_gm_scale
            log_probs = ((log_probs ** gm_scale) * (original_probs ** (1 - gm_scale)))  # + SmallConst
            log_probs = top_k_logits(log_probs, k=args.top_k, probs=True)  # + SmallConst

            if jt.sum(log_probs) <= 1:
                log_probs = log_probs / jt.sum(log_probs)
        
        else:
            # logits = top_k_logits(logits, k=args.top_k)  # + SmallConst
            log_probs = jt.nn.softmax(logits, dim=-1)
            
        #==============================================from [50256] -> [1]==============================================
        if sample:
            # likelywords = jt.topk(log_probs, k=args.top_k, dim=-1)
            # print(enc.decode(likelywords[1].tolist()[0]))
            # print(likelywords[0].tolist())
            # np.random.choice()
            #TODO
            # prev = jt.multinomial(log_probs, num_samples=1)
            prev = torch.multinomial(torch.tensor(log_probs.tolist()), num_samples=1)
            prev = jt.array(prev.tolist())
        else:
            # _, prev = jt.topk(log_probs, k=1, dim=-1)
            prev = torch.multinomial(torch.tensor(log_probs.tolist()), num_samples=1)
            prev = jt.array(prev.tolist())

        output = prev if output is None else jt.concat((output, prev), dim=1)  # update output
        print(enc.decode(output.tolist()[0]))

    return output, true_discrim_loss, loss_in_time

# %%
def perturb_past(past, model, prev, args, classifier, true_past, original_probs,stepsize=0.01, vocab_size=50257, 
                good_index=None,accumulated_hidden=None,  grad_norms=None):
    #prev: 2d jt.array
    #past: tuple(tuple(tuple(2d jt.array)))
    #==============================================use prev to query past in model==============================================
    gm_scale, kl_scale = args.fusion_gm_scale, args.fusion_kl_scale
    one_hot_vectors = []
    if good_index is not None:
        for good_list in good_index:
            good_list = list(filter(lambda x: len(x) <= 1, good_list))
            good_list = jt.array(good_list)
            num_good = good_list.shape[0]
            one_hot_good = jt.zeros(num_good, vocab_size)
            one_hot_good.scatter_(1, good_list, 1)
            one_hot_vectors.append(one_hot_good)

    # Generate inital perturbed past, we accumulate grad on this
    past_perturb_orig = []
    for layer in past:
        past_perturb_orig.append([jt.zeros_like(x).astype(jt.float32) for x in layer])

    if accumulated_hidden is None:
        accumulated_hidden = 0

    # ==============================================Generate a mask is gradient perturbated is based on a past window=============================================
    # window mask is all you need: it combines decay mask and window_length(naive window mask)
    window_length = args.window_length
    _, _, current_length, _ = past[0][0].shape
    #mask-part1-decay mask
    if args.decay:
        decay_mask = jt.arange(0., 1.0 + SmallConst, 1.0/(window_length))[1:]
    else:
        decay_mask = 1.0
    #mask-part2
    if current_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0][0].shape[:-2]) + tuple([window_length]) + tuple(
            past[0][0].shape[-1:]) #(batch_size, num_heads) + (seq_length) + (n_embd)

        zeros_key_val_shape = tuple(past[0][0].shape[:-2]) + tuple([current_length - window_length]) + tuple(
            past[0][0].shape[-1:])

        ones_mask = jt.ones(ones_key_val_shape)
        ones_mask = decay_mask*ones_mask.permute(0, 1, 3, 2)
        ones_mask = ones_mask.permute(0, 1, 3, 2)

        window_mask = jt.concat((ones_mask, jt.zeros(zeros_key_val_shape)), dim=-2)
    else:
        window_mask = jt.ones_like(past[0][0])
    #====================================perturb the past for args.num_iteration times, which is similar to do 'optimizer step' for several times=================
    def key_values_add(x,y):
            return list(map(add, x,y))
    loss_per_iter = []
    for i in range(args.num_iterations):
        perturbed_past = list(map(key_values_add, past, past_perturb_orig))
        _, _, current_length, _ = past_perturb_orig[0][0].shape
        # Compute hidden using perturbed past
        result = model(prev, past_key_values=perturbed_past)
        hidden = result['hidden_states'][-1] #(batch_size, 1, 1024)
        new_accumulated_hidden = accumulated_hidden + jt.sum(hidden, dim=1) #(batch_size, 1024)

        # TODO: Check the layer-norm consistency of this with trained discriminator
        #TODO
        logits = result['logits']
        logits = logits[:, -1, :]
        probabs = jt.nn.softmax(logits, dim=-1)
        #TODO
        #========================================================================calculate the loss================================================================
        loss = 0.0
        loss_list = []
        #bag of words
        if args.loss_type == 1 or args.loss_type == 3:
            for one_hot_good in one_hot_vectors:
                good_logits = jt.matmul(probabs, jt.transpose(one_hot_good))
                loss_word = good_logits
                loss_word = jt.sum(loss_word)
                loss_word = -jt.log(loss_word)
                #loss_word = torch.sum(loss_word) /torch.sum(one_hot_good)
                loss += loss_word
                loss_list.append(loss_word)
            print('words', loss)

        if args.loss_type == 2 or args.loss_type == 3:
            new_true_past = true_past
            #after using prev to query past, then do args.horizon_length times query, to better calculate the discriminator loss
            # for i in range(args.horizon_length):

            #     future_probabs = jt.nn.softmax(logits, dim=-1)  # Get softmax
            #     future_probabs = jt.unsqueeze(future_probabs, dim=1)
            #     _,future_input_id = jt.topk(future_probabs, k=1, dim=-1)

            #     res= model(future_input_id, past_key_values=new_true_past)
            #     new_true_past = res['past_key_values']
            #     future_hidden = res['hidden_states'][-1]  # Get expected hidden states
            #     new_accumulated_hidden = new_accumulated_hidden + jt.sum(future_hidden, dim=1)
            # predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1 + args.horizon_length))
            predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1))
            label = jt.array([args.label_class for i in range(predicted_sentiment.shape[0])], dtype=jt.int64)
            discrim_loss = nn.cross_entropy_loss(predicted_sentiment, label)
            print('discrim', discrim_loss)
            loss += discrim_loss
            loss_list.append(discrim_loss)


        kl_loss = 0.0
        if kl_scale > 0.0:
            p = (jt.nn.softmax(original_probs[:, -1:, :], dim=-1)).squeeze(dim=1) #(batch_size, 50256)
            #TODO correct
            p = p + SmallConst * (p <= SmallConst)
            correction = SmallConst * (probabs <= SmallConst)
            corrected_probabs = probabs + correction
            kl_loss = kl_scale * ((corrected_probabs * jt.log(corrected_probabs / p)).sum())
            loss += kl_loss

        print('Discrim Loss: ',(loss - kl_loss))
        loss_per_iter.append(loss)
        #========================================================================get the grad of past========================================================================
        grad = []
        for layer in perturbed_past:
            grad.append([jt.grad(loss, block, retain_graph=False) for block in layer])

        if grad_norms is not None and args.loss_type == 1:
            grad_norms = [jt.max(grad_norms[index], jt.norm(p_.grad * window_mask)) for index, p_ in
                          enumerate(past_perturb_orig)]
        else:
            grad_norms = []
            for layer in enumerate(grad):
                # grad_norms.append([(jt.norm(block * window_mask) + SmallConst) for block in layer])
                #TODO
                grad_norms.append([1 for block in layer])

        perturb_grad = []
        for i, layer in enumerate(past_perturb_orig):
            perturb_grad.append([-stepsize * (grad[i][j] * window_mask / grad_norms[i][j] ** args.gamma) for j, _ in enumerate(layer)])
        
        past_perturb_orig = list(map(key_values_add, perturb_grad, past_perturb_orig))
        # jt.sync_all()
        # jt.display_memory_info()
        # jt.gc()

    
    perturbed_past = list(map(key_values_add, past, past_perturb_orig))
    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iter

# %%
parser = argparse.ArgumentParser()
# Bags of words used for PPLM-BoW. Multiple BoWs separated by ;
parser.add_argument('--bag-of-words', '-B', type=str, default=None, 
                    help='')
# Discriminator to use for loss-type 2
parser.add_argument('--discrim', '-D', type=str, default='sentiment', 
                    choices=('clickbait', 'sentiment', 'toxicity'), 
                    help='')
parser.add_argument('--label-class', type=int, default=2, help='Class label used for the discriminator')#2-positive; 3-negative
parser.add_argument('--stepsize', type=float, default=20) #0.02 multinomial
parser.add_argument("--length", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--temperature", type=float, default=1.0)
# top-k采样
parser.add_argument("--top_k", type=int, default=10)
# 
parser.add_argument("--fusion-gm-scale", type=float, default=0.9)
parser.add_argument("--fusion-kl-scale", type=float, default=0.01)
parser.add_argument('--nocuda', action='store_true', help='no cuda')
# Generate from end-of-text as prefix
parser.add_argument('--uncond', action='store_true', help='前缀为end-of-text')
parser.add_argument("--cond-text", type=str, default='The house', help='Prefix texts to condition on')
parser.add_argument('--num-iterations', type=int, default=3)
parser.add_argument('--grad-length', type=int, default=10000)
parser.add_argument('--num-samples', type=int, default=1,
                    help='Number of samples to generate from the modified latents')
parser.add_argument('--horizon-length', type=int, default=1, help='Length of future to optimize over')
# parser.add_argument('--force-token', action='store_true', help='no cuda')
parser.add_argument('--window-length', type=int, default=0,
                    help='Length of past which is being optimizer; 0 corresponds to infinite window length')
parser.add_argument('--decay', action='store_true', help='whether to decay or not')
parser.add_argument('--gamma', type=float, default=1.5)

args = parser.parse_args(args=[])
# 设置随机种子
# np.random.seed(args.seed)
jt.core.set_seed(args.seed)

# use cuda
if not args.nocuda: 
    jt.flags.use_cuda = 0

# load_pretrained
# if args.uncond:
#     seq = [[50256, 50256]]

# else:
#     # 前缀词
#     raw_text = args.cond_text
#     while not raw_text:
#         print('Did you forget to add `--cond-text`? ')
#         raw_text = input("Model prompt >>> ")
#     seq = [[50256] + enc.encode(raw_text)]
prefix = ['The country']#, 'The chicken', 'The house', 'The food']
seq = [[50256] + enc.encode(p) for p in prefix]
# %%
collect_gen = dict()
current_index = 0 
for out in seq:

    text = enc.decode(out)
    print("=" * 40 + " Prefix of sentence " + "=" * 40)
    print(text)
    print("=" * 80)

    out1, out_perturb, discrim_loss_list, loss_in_time_list = latent_perturb(model=model, args=args, context=out)

    text_whole = enc.decode(out1.tolist()[0])

    print("=" * 80)
    print("=" * 40 + " Whole sentence (Original)" + "=" * 40)
    print(text_whole)
    print("=" * 80)

    out_perturb_copy = out_perturb

    generated = 0
    # 干扰后的结果
    for out_perturb in out_perturb_copy:
        try:
            print("=" * 40 + " Whole sentence (Perturbed)" + "=" * 40)
            text_whole = enc.decode(out_perturb.tolist()[0])
            print(text_whole)
            print("=" * 80)
        except:
            pass
        collect_gen[current_index] = [out, out_perturb, out1]
        # Save the prefix, perturbed seq, original seq for each index

        current_index = current_index + 1

origin = []
perturb = []
for idx in range(current_index):
    sen = collect_gen[idx]
    origin.append(enc.decode(sen[2].tolist()[0]))
    perturb.append(enc.decode(sen[1].tolist()[0]))


dist_score1 = dist_n(origin, 1)
dist_score2 = dist_n(origin, 2)
dist_score3 = dist_n(origin, 3)

dist_score1p = dist_n(perturb, 1)
dist_score2p = dist_n(perturb, 2)
dist_score3p = dist_n(perturb, 3)

input1 = torch.tensor(enc.encode(origin))
input2 = torch.tensor(enc.encode(perturb))

pp1 = copy_model(input1,labels=input1)
pp2 = copy_model(input2, labels=input2)
print(dist_score1, dist_score1p,pp1.loss,pp2.loss)
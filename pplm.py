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
import copy
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.optim as optim 

import jittor as jt
from style_utils import to_var, top_k_logits
from jittor.dataset import Dataset as jtDataset

from torchtext import data as torchtext_data
from torchtext import datasets
from modeling_jittor_gpt2 import GPT2PreTrainedModel, GPT2Model
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import transformers
import random


enc = transformers.GPT2Tokenizer.from_pretrained('') # default path
def sample_from_hidden(model, args, classifier, context=None, past=None,
                       sample=True, perturb=True, good_index=None):
    
    output = jt.int64(context).unsqueeze(0) if context else None
    grad_norms = None
    loss_in_time = []
    for i in trange(args.length, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        # Therefore, use everything from before current i/p token to generate relevant past

        if past is None and output is not None:
            prev = output[:, -1:]
            _, past = model(output[:, :-1])
            original_probs, true_past = model(output)
            true_hidden = model.hidden_states

        else:
            original_probs, true_past = model(output)
            true_hidden = model.hidden_states

        # Modify the past if necessary

        if i >= args.grad_length:
            current_stepsize = args.stepsize * 0
        else:
            current_stepsize = args.stepsize

        if not perturb or args.num_iterations == 0:
            perturbed_past = past

        else:
            accumulated_hidden = model.hidden_states[:, :-1, :]
            accumulated_hidden = jt.sum(accumulated_hidden, dim=1)

            perturbed_past, _, grad_norms, loss_per_iter = perturb_past(past, model, prev, args,
                                                                        good_index=good_index, stepsize=current_stepsize,
                                                                        original_probs=original_probs,
                                                                        true_past=true_past,
                                                                        accumulated_hidden=accumulated_hidden,
                                                                        classifier=classifier,
                                                                        grad_norms=grad_norms)
            loss_in_time.append(loss_per_iter)

        test_logits, past = model(prev, past=perturbed_past)
        # test_logits = F.softmax(test_logits[:, -1, :], dim=-1)
        # likelywords = torch.topk(test_logits, k=10, dim=-1)
        # print(enc.decode(likelywords[1].tolist()[0]))

        if classifier is not None:
            ce_loss = jt.nn.CrossEntropyLoss()
            predicted_sentiment = classifier(jt.mean(true_hidden, dim=1))
            label = jt.int64([args.label_class])
            true_discrim_loss = ce_loss(predicted_sentiment, label)
            print("true discrim loss", true_discrim_loss.data.cpu().numpy())
        else:
            true_discrim_loss = 0 

        hidden = model.hidden_states  # update hidden
        logits = model.forward_hidden(hidden)
        logits = logits[:, -1, :] / args.temperature  # + SmallConst

        # logits = top_k_logits(logits, k=args.top_k)  # + SmallConst

        log_probs = jt.nn.softmax(logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:

            # original_probs = top_k_logits(original_probs[:, -1, :]) #+ SmallConst
            original_probs = jt.nn.softmax(original_probs[:, -1, :], dim=-1)
            # likelywords = jt.topk(original_probs, k=10, dim=-1)
            # print(enc.decode(likelywords[1].tolist()[0]))

            gm_scale = args.fusion_gm_scale
            log_probs = ((log_probs ** gm_scale) * (original_probs ** (1 - gm_scale)))  # + SmallConst
            log_probs = top_k_logits(log_probs, k=args.top_k, probs=True)  # + SmallConst

            if jt.sum(log_probs) <= 1:
                log_probs = log_probs / jt.sum(log_probs)
        
        else:
            logits = top_k_logits(logits, k=args.top_k)  # + SmallConst
            log_probs = jt.nn.softmax(logits, dim=-1)

        if sample:
            # likelywords = jt.topk(log_probs, k=args.top_k, dim=-1)
            # print(enc.decode(likelywords[1].tolist()[0]))
            # print(likelywords[0].tolist())
            # np.random.choice()
            prev = jt.multinomial(log_probs, num_samples=1)
        else:
            _, prev = jt.topk(log_probs, k=1, dim=-1)
        # if perturb:
        #     prev = future
        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
        print(enc.decode(output.tolist()[0]))

    return output, true_discrim_loss, loss_in_time


def run_model():
    parser = argparse.ArgumentParser()
    # 预训练模型
    parser.add_argument('--model_path', '-M', type=str, default='gpt-2_pt_models/',
                        help='预训练模型')
    # Bags of words used for PPLM-BoW. Multiple BoWs separated by ;
    parser.add_argument('--bag-of-words', '-B', type=str, default=None, 
                        help='')
    # Discriminator to use for loss-type 2
    parser.add_argument('--discrim', '-D', type=str, default=None, 
                        choices=('clickbait', 'sentiment', 'toxicity'), 
                        help='')
    parser.add_argument('--label-class', type=int, default=-1, help='Class label used for the discriminator')
    parser.add_argument('--stepsize', type=float, default=0.02)
    parser.add_argument("--length", type=int, default=100)
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
    parser.add_argument("--cond-text", type=str, default='The lake', help='Prefix texts to condition on')
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

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    jt.core.set_seed(args.seed)

    # use cuda
    if not args.nocuda: 
        jt.flags.use_cuda = 1

    # load_pretrained
    model = GPT2Model()
    model.load(args.model_path)

    # eval
    model.eval()

    # 固定参数
    # TO DO
    for param in model.Partermer.item():
        pass

    if args.uncond:
        seq = [[50256, 50256]]

    else:
        # 前缀词
        raw_text = args.cond_text
        while not raw_text:
            print('Did you forget to add `--cond-text`? ')
            raw_text = input("Model prompt >>> ")
        seq = [[50256] + enc.encode(raw_text)]

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

    return

if __name__ == '__main__':
    run_model()



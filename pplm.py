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

from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import transformers
import random

# try to use cuda
jt.flags.use_cuda = 1
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
            prev = jt.multinomial(log_probs, num_samples=1)
        else:
            _, prev = jt.topk(log_probs, k=1, dim=-1)
        # if perturb:
        #     prev = future
        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
        print(enc.decode(output.tolist()[0]))

    return output, true_discrim_loss, loss_in_time


def run_model():
    pass

if __name__ == '__main__':
    run_model()


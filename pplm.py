# %%
import argparse
from tqdm import trange
from operator import add
import transformers
import jittor as jt
import gpt2
from style_utils import top_k_logits, dist_n
from classification import ClassificationHead, Discriminator2mean
from evaluate import load

SmallConst = 1e-15
enc = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')

# %%
def perturb_past(past, model, prev, args, classifier, true_past, original_probs,stepsize=0.01, vocab_size=50257, 
                good_index=None,accumulated_hidden=None,  grad_norms=None):
    gm_scale, kl_scale = args.fusion_gm_scale, args.fusion_kl_scale
    one_hot_vectors = []
    if good_index is not None:
        for good_list in good_index:
            good_list = jt.array(good_list)
            num_good = good_list.shape[0]
            one_hot_good = jt.zeros(num_good, vocab_size)
            one_hot_good.scatter_(1, good_list, jt.ones(num_good, 1))
            one_hot_vectors.append(one_hot_good)
    #Generate inital perturbed past, we accumulate grad on this
    accu_grad = []
    for layer in past:
        accu_grad.append([jt.zeros_like(x).astype(jt.float32) for x in layer])

    if accumulated_hidden is None:
        accumulated_hidden = 0

    # ==============================================Generate a mask is gradient perturbated is based on a past window=============================================
    # window mask combines decay mask and window_length(naive window mask)
    window_length = args.window_length
    _, _, current_length, _ = past[0][0].shape
    #mask-part1-decay mask
    if args.decay:
        decay_mask = jt.arange(0., 1.0 + SmallConst, 1.0/(window_length))[1:]
    else:
        decay_mask = 1.0
    #mask-part2-moving-window
    if current_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0][0].shape[:-2]) + tuple([window_length]) + tuple(
            past[0][0].shape[-1:]) #shape: (batch_size, num_heads) + (seq_length) + (n_embd)

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
        perturbed_past = list(map(key_values_add, past, accu_grad))
        _, _, current_length, _ = accu_grad[0][0].shape
        # Compute hidden using perturbed past
        result = model(prev, past_key_values=perturbed_past)
        hidden = result['hidden_states'][-1] #(batch_size, 1, 1024)
        new_accumulated_hidden = accumulated_hidden + jt.sum(hidden, dim=1) #(batch_size, 1024)

        logits = result['logits']
        logits = logits[:, -1, :]
        probabs = jt.nn.softmax(logits, dim=-1)
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
    
                loss += loss_word
                loss_list.append(loss_word)
            print('words', loss)

        if args.loss_type == 2 or args.loss_type == 3:
            new_true_past = true_past
            #after using prev to query past, then do args.horizon_length times query, to better calculate the discriminator loss
            for i in range(args.horizon_length):
                future_probabs = jt.unsqueeze(probabs, dim=1)
                embeds = jt.matmul(future_probabs, model.transformer.wte.weight)
                
                res= model(inputs_embeds= embeds, past_key_values=new_true_past)
                new_true_past = res['past_key_values']
                future_hidden = res['hidden_states'][-1]  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + jt.sum(future_hidden, dim=1)
            predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1 + args.horizon_length))
            
            label = jt.array([args.label_class for i in range(predicted_sentiment.shape[0])], dtype=jt.int64)
            discrim_loss = jt.nn.cross_entropy_loss(predicted_sentiment, label)
            if args.log_level > 0:
                print('discrim', discrim_loss)
            loss += discrim_loss
            loss_list.append(discrim_loss)


        kl_loss = 0.0
        if kl_scale > 0.0:
            p = (jt.nn.softmax(original_probs[:, -1:, :], dim=-1)).squeeze(dim=1) #(batch_size, 50256)
           
            p = p + SmallConst * (p <= SmallConst)
            correction = SmallConst * (probabs <= SmallConst)
            corrected_probabs = probabs + correction
            kl_loss = kl_scale * ((corrected_probabs * jt.log(corrected_probabs / p)).sum())
            loss += kl_loss
        if args.log_level > 0:
            print('Discrim Loss: ',(loss - kl_loss))
        loss_per_iter.append(loss)
        #========================================================================get the grad of past========================================================================
        grad = []
        for layer in perturbed_past:
            grad.append([jt.grad(loss, block, retain_graph=False) for block in layer])
        def my_norm(x):
            return (x.sqr()).sum().sqrt()
        if grad_norms is not None and args.loss_type == 1:
            grad_norms = [jt.max(grad_norms[index], my_norm(p_.grad * window_mask)) for index, p_ in
                          enumerate(accu_grad)]
        else:
            grad_norms = []
            for _,layer in enumerate(grad):
                grad_norms.append([(my_norm(block * window_mask) + SmallConst) for block in layer])

        perturb_grad = []
        for i, layer in enumerate(accu_grad):
            perturb_grad.append([-stepsize * (grad[i][j] * window_mask / grad_norms[i][j] ** args.gamma) for j, _ in enumerate(layer)])
        
        accu_grad = list(map(key_values_add, perturb_grad, accu_grad))

    perturbed_past = list(map(key_values_add, past, accu_grad))
    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iter

# %%
def sample_from_hidden(model, args, classifier, context=None, past=None,
                       sample=False, perturb=True, good_index=None):
    
    output = jt.int64(context).unsqueeze(0) if context else None
    grad_norms = None
    true_discrim_loss = []
    loss_per_word = []
    for i in trange(args.length, ascii=True):
        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        if past is None and output is not None:
            prev = output[:, -1:]
            res =  model(output[:, :-1])
            past = res['past_key_values']

        res = model(output)
        original_probs, true_past = res['logits'], res['past_key_values']
        true_hidden = res['hidden_states'][-1]

        if i >= args.grad_length:
            current_stepsize = args.stepsize * 0
        else:
            current_stepsize = args.stepsize

        #not perturb
        if not perturb or args.num_iterations == 0:
            perturbed_past = past
        #perturb
        else:
            accumulated_hidden = true_hidden[:, :-1, :] #all hidden states except query word
            accumulated_hidden = jt.sum(accumulated_hidden, dim=1)
            perturbed_past, _, grad_norms, loss_per_iter = perturb_past(past, model, prev, args,
                                                                        classifier=classifier,
                                                                        good_index=good_index, stepsize=current_stepsize,
                                                                        original_probs=original_probs,
                                                                        true_past=true_past,
                                                                        accumulated_hidden=accumulated_hidden,
                                                                        grad_norms=grad_norms)
            loss_per_word.append(loss_per_iter)

        if classifier is not None:
            ce_loss = jt.nn.CrossEntropyLoss()
            predicted_sentiment = classifier(jt.mean(true_hidden, dim=1))
            label = jt.int64([args.label_class])
            true_discrim_loss.append(ce_loss(predicted_sentiment, label))
            if args.log_level > 0:
                print("true discrim loss", true_discrim_loss[-1])
        else:
            true_discrim_loss.append(0.0)
        
        res = model(prev, past_key_values=perturbed_past)
        test_logits, past = res['logits'], res['past_key_values']
        logits = test_logits
        logits = logits[:, -1, :] / args.temperature 
        log_probs = jt.nn.softmax(logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:
            original_probs = jt.nn.softmax(original_probs[:, -1, :], dim=-1)
            gm_scale = args.fusion_gm_scale
            log_probs = ((log_probs ** gm_scale) * (original_probs ** (1 - gm_scale)))  
            log_probs = top_k_logits(log_probs, k=args.top_k, probs=True) 

            if jt.sum(log_probs) <= 1:
                log_probs = log_probs / jt.sum(log_probs)
        else:
            pert_logits = top_k_logits(logits, k=args.top_k)  
            log_probs = jt.nn.softmax(pert_logits, dim=-1)
            
        if sample:
            prev, best_ll = None, None
            for idx in range(args.sampling_nums):
                temp_prev = jt.multinomial(log_probs, num_samples=1)

                temp = temp_prev.clone() if output is None else jt.concat((output, temp_prev), dim=1)
                
                result = model(temp)
                temp_hidden = result.last_hidden_state[0]
                ll = classifier(temp_hidden.mean(dim=0))[args.label_lass]
                dist_score = (dist_n(temp,1) + dist_n(temp, 2) + dist_n(temp, 3)) / 3
                
                if (best_ll is None) or (ll > best_ll and dist_score > args.dist_threshold):
                    prev = temp
                    best_ll = ll
        else:
            prev = jt.multinomial(log_probs, num_samples=1)
            
        output = prev if output is None else jt.concat((output, prev), dim=1)  # update output
        if args.log_level > 0:
            print(enc.decode(output.tolist()[0]))

    return output, true_discrim_loss, loss_per_word

# %%
def latent_perturb(model, args, context=None, sample=True):
    if args.discrim == 'sentiment':
        classifier = ClassificationHead(class_size=5, embed_size=1024)
        classifier.load("discrim_models/sentiment_classifierhead.pt")
        classifier.eval()
        if args.label_class < 0:
            raise Exception('Wrong class for sentiment, use --label-class 2 for *very positive*, 3 for *very negative*')
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
        if args.log_level > 0:
            print('Both PPLM-BoW and PPLM-Discrim are on. This is not optimized.')
        args.loss_type = 3

    elif args.bag_of_words:
        args.loss_type = 1
        if args.log_level > 0:
            print('Using PPLM-BoW')

    elif classifier is not None:
        args.loss_type = 2
        if args.log_level > 0:
            print('Using PPLM-Discrim')
    else:
        raise Exception('Supply either --bag-of-words (-B) or --discrim -D')
    #==================================================generate non perturbed words=======================================================
    original, _, _ = sample_from_hidden(model=model, args=args, context=context,
                                  perturb=False, good_index=good_index, classifier=classifier)

    #==================================================generate perturbed words===========================================================
    perturbed_list = []
    discrim_loss_list = []
    loss_per_sen = []

    for i in range(args.num_samples): #another implementation of resample (with obvious improvement), but much higher computation cost, so usually set num_samples=1
        perturbed, discrim_loss, loss_per_word = sample_from_hidden(model=model, args=args, context=context,
                                                        perturb=True, good_index=good_index,
                                                         classifier=classifier)
        perturbed_list.append(perturbed)
        if classifier is not None:
            discrim_loss_list.append(discrim_loss)
        loss_per_sen.append(loss_per_word)
    
    return original, perturbed_list, discrim_loss_list, loss_per_sen

# %%
config = gpt2.GPT2Config()
model = gpt2.GPT2LMHeadModel(config)
model.load('gpt2-medium.pkl')

# %%
parser = argparse.ArgumentParser()
#choose -B or -D
parser.add_argument('--bag-of-words', '-B', type=str, default=None, 
                    help='Bags of words used for PPLM-BoW. Multiple BoWs separated by ;')
parser.add_argument('--discrim', '-D', type=str, default='sentiment')
parser.add_argument('--label-class', type=int, default=2, help='Class label used for the discriminator')
#BR
parser.add_argument('--stepsize', type=float, default=0.03, help='the lr for updating past-key-values')
parser.add_argument('--num-iterations', type=int, default=0, help='iterations for updating past-key-values')
parser.add_argument('--horizon-length', type=int, default=1, help='Length of future to optimize over')
parser.add_argument('--grad-length', type=int, default=10000, help='the upper bound for times of updating gradients')
parser.add_argument('--window-length', type=int, default=0,
                    help='Length of past which is being optimizer; 0 corresponds to infinite window length')
parser.add_argument('--decay', action='store_true', help='whether to decay or not')
parser.add_argument('--gamma', type=float, default=1.0)
#BC
parser.add_argument('--sampling_nums', type=int, default=1)
parser.add_argument('--dist_threshold', type=float, default=0.8)#0.9
#general settings
parser.add_argument("--length", type=int, default=15, help='length of the generated sentences(exclude the prefix)')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--fusion-gm-scale", type=float, default=0.95)
parser.add_argument("--fusion-kl-scale", type=float, default=0.01)
parser.add_argument('--log_level',type=int, default=0,help='0 for no log, 1 for log')
parser.add_argument('--num-samples', type=int, default=1,
                    help='Number of samples to generate from the modified latents')

args = parser.parse_args(args=[])

# 设置随机种子
jt.core.set_seed(args.seed)

# use cuda
jt.flags.use_cuda = jt.has_cuda

prefix = ['Once upon a time', 'The book', 'The chicken', 'The city'
          ,'The country', 'The horse', 'The lake', 'The last time'
          ,'The movie', 'The painting', 'The pizza', 'The potato',
          'The president of the country', 'The road', 'The year is 1910.']

seq = [[50256] + enc.encode(p) for p in prefix]

# %%
collect_gen = dict() #record the output
current_index = 0 
for idx, out in enumerate(seq):
    print('Current index {}, current / all = {}%'.format(idx, idx / len(seq) * 100))
    if args.log_level > 0:
        text = enc.decode(out)
        print("=" * 40 + " Prefix of sentence " + "=" * 40)
        print(text)
        print("=" * 80)

    origin_tokens, perturb_tokens, discrim_loss_list, loss_in_time_list = latent_perturb(model=model, args=args, context=out)

    origin_output = enc.decode(origin_tokens.tolist()[0])
    if args.log_level > 0:
        print("=" * 80)
        print("=" * 40 + " Whole sentence (Original)" + "=" * 40)
        print(origin_output)
        print("=" * 80)

    if args.num_samples > 1:
        sen_discrim_loss = []
        for sen in loss_in_time_list:
            sen_loss = [jt.concat(x).mean() for x in sen]
            sen_discrim_loss.append(jt.concat(sen_loss).mean())
        sen_discrim_loss = jt.concat(sen_discrim_loss)
        index ,_ = sen_discrim_loss.argmax(dim=0)
        index = int(index.data)
        perturb_tokens = [perturb_tokens[index]]

    # generated = 0
    # 干扰后的结果
    for perturb_sen in perturb_tokens:
        if args.log_level > 0:
            print("=" * 40 + " Whole sentence (Perturbed)" + "=" * 40)
            perturb_output = enc.decode(perturb_sen.tolist()[0])
            print(perturb_output)
            print("=" * 80)
        collect_gen[current_index] = [out, perturb_tokens, origin_tokens]
        current_index = current_index + 1

# %%
origin = []
perturb = []
for idx in range(current_index):
    sen = collect_gen[idx]
    origin.append(sen[2].tolist()[0])
    perturb.append(sen[1][0].tolist()[0])
#dist-n
dist_score1 = dist_n(origin, 1)
dist_score2 = dist_n(origin, 2)
dist_score3 = dist_n(origin, 3)

dist_score1p = dist_n(perturb, 1)
dist_score2p = dist_n(perturb, 2)
dist_score3p = dist_n(perturb, 3)

print('origin: dist-1 {}, dist-2 {}, dist-3 {}\n'.format(dist_score1, dist_score2, dist_score3))
print('perturbed: dist-1 {}, dist-2 {}, dist-3 {}\n'.format(dist_score1p, dist_score2p, dist_score3p))

#sentiment accuracy
sentiment_pipeline = transformers.pipeline("sentiment-analysis")
data= []
for sen in origin:
    data.append(enc.decode(sen)[1:])
predict = sentiment_pipeline(data)
count = 0
length = len(origin)
for idx in range(length):
    if predict[idx]['label'] == 'POSITIVE':
        count += 1
origin_posi = count/length
print('origin positive rate = ',origin_posi)
print('origin negative rate = ',1 - origin_posi)
data= []
for sen in perturb:
    data.append(enc.decode(sen)[1:])
predict = sentiment_pipeline(data)
count = 0
length = len(perturb)
for idx in range(length):
    if predict[idx]['label'] == 'POSITIVE':
        count += 1
perturb_posi = count/length
print('perturb positive rate = ',perturb_posi)
print('perturb negative rate = ',1 - perturb_posi)

#perplexity
perplexity = load("perplexity", module_type="metric")
origin_decoded = [enc.decode(sen) for sen in origin]
perturb_decoded = [enc.decode(sen) for sen in perturb]
origin_ppl = perplexity.compute(predictions=origin_decoded, model_id='gpt2')
perturb_ppl = perplexity.compute(predictions=perturb_decoded, model_id='gpt2') 
print('origin mean perplexity: {}, perturbed mean perplexity: {}'.format(origin_ppl['mean_perplexity'],perturb_ppl['mean_perplexity']))



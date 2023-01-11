import jittor as jt
import numpy as np

def dist_n(sentences, n, encoded = True):
    total = 0
    for sen in sentences:
        if not encoded:
            sen = sen.split()
        grams = []
        for x in range(0, len(sen)-n+1):
            grams.append(tuple(sen[x:(x+n)]))
        total += len(set(grams)) / len(grams)
    return total / len(sentences)


def top_k_logits(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = jt.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return jt.where(logits < batch_mins, jt.ones_like(logits) * 0.0, logits)
        return jt.where(logits < batch_mins, jt.ones_like(logits) * -1e10, logits)

def multinomial(input: jt.Var, num_samples: int, replacement=False):
    input = input.numpy()
    # 检测是否有非法元素
    if num_samples <= 0:
        raise RuntimeError("cannot sample n_sample <= 0 samples")
    # 归一化
    input = input / input.sum(-1, keepdims=True)
    shape = input.shape

    input = input.reshape(-1, shape[-1])
    _shape = input.shape[:-1] + (num_samples,)
    # 采样(非并行)
    output = np.empty(shape=_shape, dtype=np.int64)
    for i in range(input.shape[0]):
        prob = input[i]
        zero_case = np.where(prob == 0)[0]
        _size = shape[-1] - len(zero_case)
        if _size < num_samples and not replacement:
            x = np.concatenate((np.random.choice(shape[-1], size=_size, replace=replacement, p=prob), zero_case[0: num_samples - _size]), axis=0)
            
        else:
            x = np.random.choice(shape[-1], size=num_samples, replace=replacement, p=prob)
        output[i] = x

    shape = shape[:-1] + (num_samples,)
    output = jt.Var(output.reshape(shape))
    return output

        

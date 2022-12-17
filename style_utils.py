import jittor as jt
from jittor import Function
# from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False):
    pass

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

class multinomial(Function):
    def execute(self, input, num_samples, replacement=False):
        return jt.code(shape=input.shape, dtype='int64', inputs=[input, num_samples, replacement], 
            cpu_src='''
                for (int i=0; i<)
            '''
        )

        

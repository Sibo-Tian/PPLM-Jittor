import jittor as jt
import gpt2

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
    def __init__(self, class_size=5, embed_size=1024, head=None, model=None, cache_address='gpt2-medium.pkl'):
        super().__init__()
        if head is None:
            self.classifierhead = ClassificationHead(class_size=class_size, embed_size=embed_size)
        else:
            self.classifierhead = head
        
        if model is None:
            config = gpt2.GPT2Config()
            self.model = gpt2.GPT2LMHeadModel(config)
            self.model.load(cache_address)
        else:
            self.model = model
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
        hidden = hidden * mask_src  
        hidden = hidden.permute(0, 2, 1)
        
        x =  jt.sum(hidden, dim=1)/(jt.sum(mask_src, dim=-1) + 1e-10)
        x = self.classifierhead(x)
        x = jt.nn.log_softmax(x, dim=-1)
        return x
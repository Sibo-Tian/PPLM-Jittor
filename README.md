# PPLM-Jittor

### PPLM

pplm.ipynb & pplm.py are totally the same. They are for sentence generation.

### Train discriminator

train.ipynb & train.py are totally the same. They are for discriminator training.

### Backbone - GPT2

gpt2.py is the Jittor GPT2. As the checkpoint for GPT2 is too large, you can load pytorch checkpoint like this

```
src = transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium')
dest = gpt2.GPT2LMHeadModel(config)
dest.load_state_dict(copy_model.state_dict())
```


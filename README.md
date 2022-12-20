# PPLM-Jittor

Implementation of PPLM based on Jittor

### 12.21 update

1. train.ipynb | train.py是训练classifier的脚本，二者完全相同；目前只支持sentiment的数据集，后续需要进一步填空（填空两个pass语句）；
2. gpt2.py是全部的jittor gpt2的代码
3. final.ipynb | final.py是pplm的脚本，二者完全相同，目前可以在不perturb的情况下，正常生成；

### 现在的问题

* [ ] jittor的gpt进行一次生成的速度极慢，远远慢于pytorch，目前尚未发现原因；
* [ ] perturb中的求导仅能完成几次，后续会报错，且无法解决；

#### TODO

* [ ] 总结pre中我们需要做哪些实验：必做的实验的设定是什么，实验是生成怎样的语句等等
* [ ] 进行上述实验
* [ ] 训练classifier，三个数据集

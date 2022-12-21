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

    A：根据此前大作业的讲解要求，我们**必须完成的实验**是discriminator attribute model（**PPLM-discriminator**）:根据SST-5数据集完成对sentiment任务的**BCR**（即在过程中**更新梯度$\Delta \tilde{H_t}$**，根据top-k**采样r次**并根据**对数似然得分**选取最佳样本）学习，完成**Perplexity**（衡量流利度）和**Dist-1 2 3**（衡量多样性）指标的度量。其余数据集实验可以等之后进行补充与完善，但不是必做内容。具体实验内容与表格见论文第8页[4.3节](https://readpaper.com/pdf-annotate/note?pdfId=4518414770817884161&noteId=1568028772593421056)
    
* [ ] 进行上述实验
* [ ] 训练classifier，三个数据集

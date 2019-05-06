Code for the paper [An Attention-Based Word-Level Interaction Model for Knowledge Base Relation Detection](https://ieeexplore.ieee.org/document/8546730). 


### Requirements
- GPU and CUDA 8 are required
- python >=3.5 
- pytorch 0.4.1 # we used this version, other verion may also work but not tested.   
- pandas
- msgpack
- spacy 1.x
- cupy
- pynvrtc == 8

# Data
## Relation detection data sets (SimpleQuestions and WebQSP)
download the datasets into ./dat
```
cd ./dat
git clone https://github.com/Gorov/KBQA_RE_data.git 
``` 

## Preparing the vocab and word vectors
The word vectors extracted for these two data sets are uploaded. 
```
./dat/SimpleQA_vocab.txt
./dat/WebQA_vocab.txt
```
The scripts needed for reproduction is also available. 
Firstly, download and unzip [glove.6B.300d.txt](http://nlp.stanford.edu/data/wordvecs/glove.6B.zip) and unzip it into  `./dat`. 

Secondly, run 
```
python ./ABWIMSimpleQA/word2vec_prepare_for_simpleQA.py
python ./ABWIMWebQSP/word2vec_prepare_for_webQA.py
# These two scripts are not tested recently, please raise an issue if any question encountered.
``` 


## Training
On SimpleQuestions, run 
```
cd ./ABWIMSimpleQA
python train_abwim.py
```

The results should be like 
```
05/05/2019 07:10:03 ================================= epoch: 31 ==================================
05/05/2019 07:10:03 loss=0.010898256674408913
05/05/2019 07:10:03 train_loss_not_null=7849.0, total_train_num=205056,
05/05/2019 07:10:04 new best valid f1 found
05/05/2019 07:10:04 model saved to models/New_exp_idx_abwim_on_1/model_epoch_31.h5.
05/05/2019 07:10:04 valid, r_f1=0.9389853526045203, batch_num=38333, need
05/05/2019 07:10:08 test, r_f1=0.9365811053423262, batch_num=38333, need
...
05/05/2019 07:20:33 ================================= epoch: 47 ==================================
05/05/2019 07:20:33 loss=0.010160243138670921
05/05/2019 07:20:33 train_loss_not_null=7328.0, total_train_num=204800,
05/05/2019 07:20:34 new best valid f1 found
05/05/2019 07:20:34 model saved to models/New_exp_idx_abwim_on_1/model_epoch_47.h5.
05/05/2019 07:20:34 valid, r_f1=0.9394703656998739, batch_num=51150, need
05/05/2019 07:20:38 test, r_f1=0.9368722402833713, batch_num=51150, need  
...
05/05/2019 07:34:20 ================================= epoch: 68 ==================================
05/05/2019 07:34:20 loss=0.009640091098845005
05/05/2019 07:34:20 train_loss_not_null=6933.0, total_train_num=204800,
05/05/2019 07:34:21 new best valid f1 found
05/05/2019 07:34:21 model saved to models/New_exp_idx_abwim_on_1/model_epoch_68.h5.
05/05/2019 07:34:21 valid, r_f1=0.9398583761761568, batch_num=67952, need
05/05/2019 07:34:25 test, r_f1=0.9369207627735455, batch_num=67952, need  
```
Please refer to `./ABWIMSimpleQA/logs/New_exp_idx_abwim_on_1.log` for detailed training log.

On WebQSP, run 
```
cd ./ABWIMWebQSP
python train_abwim.py
```
The results should be like 
```
05/06/2019 10:15:15 ================================= epoch: 31 ==================================
05/06/2019 10:15:15 loss=0.0031128262635320425
05/06/2019 10:15:15 train_loss_not_null=815.0, total_train_num=67520,
3.8928608894348145 inference time
05/06/2019 10:15:19 new best valid f1 found
05/06/2019 10:15:19 model saved to models/New_exp_idx_webqsp_on_1/model_epoch_-1.h5.
05/06/2019 10:15:19 r_f1=0.8671922377198302
selected indices=1414, review indices=66220
remove item 0
05/06/2019 10:15:32 batch_idx=1000, loss=0.0031618636567145586, adv_loss=0
13.921881675720215 training a epoch
05/06/2019 10:15:33 ================================= epoch: 32 ==================================
05/06/2019 10:15:33 loss=0.00302296900190413
05/06/2019 10:15:33 train_loss_not_null=842.0, total_train_num=67584,
3.8871352672576904 inference time
05/06/2019 10:15:37 new best valid f1 found
05/06/2019 10:15:37 model saved to models/New_exp_idx_webqsp_on_1/model_epoch_-1.h5.
05/06/2019 10:15:37 r_f1=0.8677986658580958
selected indices=1422, review indices=66219
remove item 0
05/06/2019 10:15:50 batch_idx=1000, loss=0.0030801771208643913, adv_loss=0
```
Please refer to `./ABWIMWebQSP/logs/New_exp_idx_abwim_on_1.log` for detailed training log.

 

 


### Credits
Autor of sru: [Tao Lei](https://github.com/taolei87/sru).

Author of the Document Reader model: [Danqi Chen](https://github.com/danqi).

Thanks to [Yu et al.](http://arxiv.org/abs/1704.06194) for releasing the data sets.  


 
 
  
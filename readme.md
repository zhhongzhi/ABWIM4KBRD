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
```
Please refer to `./ABWIMSimpleQA/logs/New_exp_idx_abwim_on_1.log` for detailed training log.

On WebQSP, run 
```

```
 

 


### Credits
Autor of sru: [Tao Lei](https://github.com/taolei87/sru).

Author of the Document Reader model: [Danqi Chen](https://github.com/danqi).

Thanks to [Yu et al.](http://arxiv.org/abs/1704.06194) for releasing the data sets.  


 
 
  
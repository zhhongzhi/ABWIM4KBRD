Code for the paper [An Attention-Based Word-Level Interaction Model for Knowledge Base Relation Detection](https://ieeexplore.ieee.org/document/8546730). 

# Data
## Relation detection data sets (SimpleQuestions and WebQSP)
download the datasets into ./dat
```
cd ./dat
git clone https://github.com/Gorov/KBQA_RE_data.git
``` 

## Preparing the vocab and word vectors
The word vectors extracted for these two data sets are uploaded. 

If you wanna to reproduce the extraction procedure, please refer to the scripts  
`./ABWIMSimpleQA/word2vec_prepare_for_simpleQA.py` and ` `. 
Firstly, download and unzip glove.6B.300d.txt from  
http://nlp.stanford.edu/data/wordvecs/glove.6B.zip 
and unzip it into  `./dat`.
Secondly, run `python3 ./ABWIMSimpleQA/word2vec_prepare_for_simpleQA.py`
and `python3 `. 
These two scripts are not tested recently, raise an issue if there is any question.

 
 
  
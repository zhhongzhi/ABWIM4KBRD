# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""
import sys
import numpy as np
# np.random.seed(0)

class VocabWithoutEmb(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with
          most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self.PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        # self.UNKNOWN_TOKEN = '#OOV#'
        self.UNKNOWN_TOKEN = '<UNK>'
        self.DSStart = 'DSStart'
        self.DSEnd = 'DSEnd'
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [self.PAD_TOKEN, self.UNKNOWN_TOKEN, self.DSStart, self.DSEnd]:   # 如果#OOV#没有在预训练的词典中，这里要加入OOV的
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            # self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))
        if sys.version[0] == '3':
            with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
                ws = [ln.strip('\n') for ln in vocab_f.readlines()]
        else:
            with open(vocab_file, 'r') as vocab_f:
                ws = [ln.strip('\n').decode('utf-8') for ln in vocab_f.readlines()]
        for w in ws:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # if emb_value_file is not None:
        #     with open(emb_value_file, 'r') as emb_value_f:   # , encoding='utf-8'
        #         for ln in emb_value_f.readlines():
        #             v = [float(v) for v in ln.strip('\n').split(',')]
        #             self.vs.append(v)
        # else:
            # self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))
        # assert len(self.vs) == len(self._word_to_id), '词嵌入的维度和词典的维度应该一致的'

        print(len(self._id_to_word))
        print(u"Vocab: Finished constructing vocabulary of {} total words.".format(self._count))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def seqword2id(self, word_seq):
        return [self.word2id(w) for w in word_seq]

    def id2seqword(self, ids, rm_padding=True):
        return [self.id2word(idx) for idx in ids if not (idx == 0 and rm_padding)]


class WordVocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""
    @staticmethod
    def load_glove(dim=300):
        f = open(u'/mydata/datasets_数据集/glove.6B.{}d.txt'.format(dim))
        glove = {}
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) < dim + 1:
                continue
            else:
                glove[tokens[0]] = map(float, tokens[1:])
        return glove

    def __init__(self, vocab_file, emb_dim, with_glove=False):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self.PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        # self.UNKNOWN_TOKEN = '#OOV#'
        self.UNKNOWN_TOKEN = '<UNK>'
        self.RELSEP = '#rel_sep#'
        self.RELPAD = '####relpad###'
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
        self.emb_dim = emb_dim
        if with_glove:
            glove = self.load_glove()

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        self.vs = []
        num = 0
        for w in [self.PAD_TOKEN, self.UNKNOWN_TOKEN, self.RELSEP, self.RELPAD]:   # 如果#OOV#没有在预训练的词典中，这里要加入OOV的
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            if with_glove and w in glove:  # 没有用glove来初始化吗？ 为什么不用呢？
                self.vs.append(glove[w])
                num += 1
            else:
                if w == self.PAD_TOKEN:
                    self.vs.append(0 * np.random.uniform(-0.5, 0.5, self.emb_dim))
                else:
                    self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))
        if sys.version[0] == '3':
            with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
                lns = [ln.strip('\n') for ln in vocab_f.readlines()]

        else:
            with open(vocab_file, 'r') as vocab_f:
                lns = [ln.strip('\n').decode('utf-8') for ln in vocab_f.readlines()]

        for ln in lns:
            w, vs = ln.split('\t')
            vs = [float(v) for v in vs.split(',')]
            if len(w) == 0 or len(vs) != emb_dim:
                continue
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            self.vs.append(np.array(vs))

        # print('inited num={}'.format(num))
        assert len(self.vs) == len(self._word_to_id), '词嵌入的维度和词典的维度应该一致的'

        self.vs = np.asarray(self.vs)
        print(u"WordVocab: Finished constructing vocabulary of {} total words.".format(self._count))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def seqword2id(self, word_seq):
        return [self.word2id(w) for w in word_seq]

    def id2seqword(self, ids, rm_padding=True):
        return [self.id2word(idx) for idx in ids if not (idx == 0 and rm_padding)]


class CharVocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, emb_dim):
        """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

        Args:
          most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self.PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
        # self.UNKNOWN_TOKEN = '#OOV#'
        self.UNKNOWN_TOKEN = '<UNK>'
        self.DSStart = 'DSStart'
        self.DSEnd = 'DSEnd'
        self.DSStartP = 'DSStartP' # P for paraphrased
        self.DSEndP = 'DSEndP'
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab
        self.emb_dim = emb_dim

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        self.vs = []
        for w in [self.PAD_TOKEN, self.UNKNOWN_TOKEN, self.DSStart, self.DSEnd, self.DSStartP, self.DSEndP]:   # 如果#OOV#没有在预训练的词典中，这里要加入OOV的
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))  # 改了这里
        if sys.version[0] == '3':
            with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
                ws = [ln.strip('\n') for ln in vocab_f.readlines()]
        else:
            with open(vocab_file, 'r') as vocab_f:
                ws = [ln.strip('\n').decode('utf-8') for ln in vocab_f.readlines()]
        ws = [w for w in ws if len(w) > 0]
        for w in ws:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1
            self.vs.append(np.random.uniform(-0.5, 0.5, self.emb_dim))

        assert len(self.vs) == len(self._word_to_id), '词嵌入的维度和词典的维度应该一致的'

        self.vs = np.asarray(self.vs)
        print(u"CharVocab: Finished constructing vocabulary of {} total words.".format(self._count))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def seqword2id(self, word_seq):
        return [self.word2id(w) for w in word_seq]

    def id2seqword(self, ids, rm_padding=True):
        return [self.id2word(idx) for idx in ids if not (idx == 0 and rm_padding)]


if __name__ == '__main__':
    pass

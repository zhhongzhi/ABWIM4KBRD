# -*- encoding: utf-8 -*-
"""

看看都有什么词  从glove加载一下词向量   然后就可以进行分类了
这次的工作是从glove中进行加载，special tokens 在构建Vocabulary对象的时候再进行随机初始化吧。

"""
import numpy as np

import dat.file_pathes as dt_p


class PrepareEmbedding:
    def __init__(self):
        pass

    def get_wq_vocab(self):
        vocab = set()
        fs = [u'{}WebQSP.RE.{}.with_boundary.withpool.dlnlp.txt'.format(dt_p.web_q_f_dp, i) for i in ['train', 'test']]
        rlt_rels = set()
        for f in fs:
            gold_rel, cdt_rels, qs = self.parse_f(f)
            for r in gold_rel:
                if isinstance(r, list):
                    rlt_rels.update(r)
                else:
                    rlt_rels.add(r)
            for cdt_rel in cdt_rels:
                rlt_rels.update(cdt_rel)
            for q in qs:
                vocab.update(q.split())

        web_q_rels = [ln.strip('\n') for ln in open(dt_p.web_qa_rel_f).readlines()]
        for rel_idx in rlt_rels:
            raw_rels = web_q_rels[rel_idx].split('..')
            for raw_rel in raw_rels:
                for sb_rel in raw_rel.split('.'):
                    vocab.update(sb_rel.split('_'))
                vocab.add('/' + raw_rel.replace('.', '/'))
        return vocab

    @staticmethod
    def parse_f(fnm):
        """
        解析simpleQuestion数据集
        :param fnm:
        :return:
        """
        lns = [ln.strip('\n') for ln in open(fnm).readlines()]
        gold_rel = []
        cdt_rels = []
        qs = []
        print(len(lns), fnm)
        for ln in lns:
            gr, cdt_rs, q = ln.split('\t')
            qs.append(q)
            if len(gr.split()) > 1:
                multi_gold_r = []
                for r in gr.split():
                    multi_gold_r.append(int(r)-1)   # 这里写错啦   简直不明白为什么还会有50%的正确率   后边不都是错误了吗？
                gold_rel.append(multi_gold_r)
            else:
                gold_rel.append(int(gr) - 1)

            if cdt_rs == 'noNegativeAnswer':
                cdt_rels.append([])
            else:
                cdt_rels.append([int(i) - 1 for i in cdt_rs.split()])
        return gold_rel, cdt_rels, qs

    @staticmethod
    def load_glove(dim=300):
        f = open(dt_p.dp + u'/glove.6B.{}d.txt'.format(dim))
        glove = {}
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) < dim + 1:
                continue
            else:
                glove[tokens[0]] = map(float, tokens[1:])
        return glove

    def init_word_emb(self):
        dim = 300
        glove = self.load_glove(dim)
        vocab = self.get_wq_vocab()
        vecs = []
        o_glove = 0
        rel_o_glove = 0
        in_glov_num = 0

        for w in vocab:
            if w in glove:
                vecs.append(glove[w])
                in_glov_num += 1
            else:
                # relation 的表示  取整个的序列的词的叠加  还是取最后的呢？  都试一下？
                if '/' in w:
                    rel_o_glove += 1
                    v = np.random.uniform(-0.5, 0.5, dim)
                    vecs.append(v)
                else:
                    vecs.append(np.random.uniform(-0.5, 0.5, dim))   #
                o_glove += 1
        print('init word embedding: total {} words, in_glov_words {} oov {} words, relation oov {}'
              ''.format(len(vocab), in_glov_num, o_glove, rel_o_glove))
        # 3116 /mydata/wp/SimpleQAMine/KBQA_RE_data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt
        # 1649 /mydata/wp/SimpleQAMine/KBQA_RE_data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt
        # init word embedding: total 7522 words, in_glov_words 3541 oov 3981 words, relation oov 3692
        # 还给初始化了很大一部分 good
        self.write_to_file(vocab, vecs)

    def write_to_file(self, vocab, emb_vs):
        lns = []
        for w, vs in zip(vocab, emb_vs):
            ln = u'{}\t{}\n'.format(w, u','.join([str(v) for v in vs]))
            lns.append(ln)
        open(dt_p.webqa_vocab_f, 'w').writelines(lns)


if __name__ == '__main__':
    PrepareEmbedding().init_word_emb()

# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""
from collections import defaultdict
import random

from tqdm import tqdm
import numpy as np
import torch

import word_vocab
import dat.file_pathes as dt_p
from ABWIM.contextualized_rep import elmo, batch_to_ids

elmo.to('cuda')

class WebQADataManager:
    def __init__(self, opt, logger):
        self.debug = False
        self.opt = opt
        self.logger = logger
        self.vocab = word_vocab.WordVocab(dt_p.webqa_vocab_f, emb_dim=300)
        q_fs = [u'{}WebQSP.RE.{}.with_boundary.withpool.dlnlp.txt'.format(dt_p.web_q_f_dp, i) for i in 'train test'.split()]

        self.q_fs = q_fs
        self.rels = [ln.strip('\n') for ln in open(dt_p.web_qa_rel_f).readlines()]
        self.elmo_of_all_rels = self.get_elmo_of_all_rels()
        self.rel_alias_words = {}
        self.rel_questions = self.get_query_of_rel()
        self.test_smp_pairs_qq, self.test_smps_qr, self.cdt_num_of_test_dat, self.gold_rel_num_test_dat, self.groups_of_rels_cnt,\
            self.cdt_rels_of_q = self.a_f2x_valid_and_test(q_fs[1])
        self.smps_rr, self.smps_rq, self.smps_qr, self.smps_qq = self.a_f2x(self.q_fs[0])

    def get_query_of_rel(self):
        rel_questions = defaultdict(list)

        # for rel_idx, rel in enumerate(self.rels):
        #     rel_questions[rel_idx].append(u' '.join(self.words_of_web_rel(rel_idx)))
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[0])
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        for r, qs in rel_questions.items():
            rel_questions[r] = list(set(qs))
        return rel_questions
        # rel_questions_sorted = {}
        # for gr, questions in rel_questions.items():
        #     r_rel = self.rels[gr]
        #     rel_questions_sorted[gr] = self.rank_alias_qs(gr, questions)
        # return rel_questions_sorted

    def calculate_idf(self):
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[0])
        qs = [q for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs))]
        w_df = defaultdict(int)
        for q in qs:
            for w in q.split():
                w_df[w] += 1
        w_idf = {w: np.log(len(qs) / (4 + df)) for w, df in w_df.items()}
        return w_idf

    def rank_alias_qs(self, rel_idx, questions):
        rel_ws = set(self.words_of_web_rel(rel_idx))
        vs = []
        questions = list(set(questions))
        for q in questions:
            q_ws = q.split()
            v = sum([1 for w in q_ws if w in rel_ws])
            v += 1e-4 * len(q_ws)
            vs.append(v)
        indexes = sorted(range(len(vs)), key=lambda k: vs[k])
        sorted_qs = [questions[idx] for idx in indexes]
        return sorted_qs

    # ######################################### trans to idx ################################################

    def words_of_web_rel(self, rel_idx):
        """
        多个地方会用到，万一要改比较方便
        """
        rel = self.rels[rel_idx]
        ws = []
        rels = []
        for sub_rel in rel.split('..'):
            tmp_sub_rel = '/' + sub_rel.replace('.', '/')
            rels.append(tmp_sub_rel)
            ws += self.words_of_a_rel(tmp_sub_rel)

        while len(rels) < 2:
            rels = [self.vocab.RELPAD] + rels
        if len(rels) > 2:
            rels = rels[:2]
        return ws + rels

    # def emb_of_seq(self, list_of_words):

    def get_elmo_of_all_rels(self):
        elmo_of_all_rels = []
        with torch.no_grad():
            for idx, rel in tqdm(enumerate(self.rels)):
                words_of_rel = self.words_of_web_rel(idx)
                words = words_of_rel[:-2]
                rels = words_of_rel[-2:]
                words = batch_to_ids([words]).cuda()
                rels = batch_to_ids([[i] for i in rels]).cuda()

                words_emb = elmo(words)['elmo_representations'][0].squeeze(0)
                rels_emb = elmo(rels)['elmo_representations'][0].squeeze(1)
                the_emb_tensor = torch.cat([words_emb, rels_emb], dim=0)
                assert the_emb_tensor.size(0) == len(words_of_rel)
                the_emb = the_emb_tensor.data.cpu().numpy()
                elmo_of_all_rels.append(the_emb)
        return elmo_of_all_rels

    def get_q_elmo_rep(self, q_ws):
        q_ws = ['<S>'] + q_ws[1:-1] + ['</S>']
        with torch.no_grad():
            q_ws_ids = batch_to_ids([q_ws]).cuda()
            q_emb = elmo(q_ws_ids)['elmo_representations'][0].squeeze(0).data.cpu().numpy()
            assert q_emb.shape[0] == len(q_ws)
            return q_emb

    @staticmethod
    def words_of_a_rel(rel):
        """
        多个地方会用到，万一要改比较方便
        """
        return rel.split('/')[-1].split('_')

    def idxs_of_rel(self, rel_idx):
        """

        :param rel_idx: 在self.rels中的编号
        :return:
        """
        return self.vocab.seqword2id(self.words_of_web_rel(rel_idx))

    def question_of_the_rel_random(self, rel, tmp_question=None):
        questions = self.rel_questions.get(rel, [])
        if len(questions) == 0:
            return ''
        if len(questions) == 1 and tmp_question == questions[0]:
            return ''
        q = random.choice(questions)
        while q == tmp_question:
            q = random.choice(questions)
        return q

    def all_question_of_the_rel(self, rel, tmp_question=None):
        questions = self.rel_questions.get(rel, [])
        return questions

    def rep_of_question(self, q):
        return self.vocab.seqword2id(q.split())

    def a_f2x_valid_and_test(self, f):
        """
        改成基于评分排序的
        :param f:
        :return:
        """
        gold_rel, cdt_rels, qs = self.parse_f(f)
        smps_qq = []
        smps_qr = []

        cdt_num_of_valid_dat = []
        gold_rel_num_q = []
        groups_of_rels_cnt = []
        test_gold_rels = []
        cdt_rels_of_q = []

        all_null_rels = []
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            grs = (gr if isinstance(gr, list) else [gr])
            test_gold_rels.append(grs)
            neg_cdts = [r for r in set(cdt_rs) if r not in grs]
            neg_cdts = sorted(neg_cdts)
            q_cdt_rels = grs + neg_cdts
            flags = ['g'] * len(grs) + ['n'] * len(neg_cdts)
            q_x = self.rep_of_question(q)
            q_c_emb = self.get_q_elmo_rep(q.split())

            groups = []
            g_rels_q_num = 0
            cdt_q_num = 0
            tmp_cdt_rels = []
            assert len(q_cdt_rels) == len(flags)
            null_rels = []
            for r, flag in zip(q_cdt_rels, flags):
                flag = 1 if flag == 'g' else 0
                smps_qr.append((q_x, self.idxs_of_rel(r), flag, q_c_emb, self.elmo_of_all_rels[r]))
                questions = self.rel_questions.get(r, [])[:self.opt.valid_q_max_num]
                if len(questions) == 0:
                    null_rels.append(r)
                if flag == 'g':
                    g_rels_q_num += len(questions)
                for cdt_q in questions:
                    smps_qq.append((q_x, self.rep_of_question(cdt_q), 1 if flag == 'g' else 0))
                cdt_q_num += len(questions)
                groups.append(len(questions))
                tmp_cdt_rels.append(r)
            if self.debug:
                if len(smps_qr) > 500:
                    break
            cdt_rels_of_q.append(tmp_cdt_rels)
            groups_of_rels_cnt.append(groups)
            cdt_num_of_valid_dat.append(cdt_q_num)
            gold_rel_num_q.append(g_rels_q_num)
            all_null_rels.append(null_rels)
        print(len(smps_qq), sum(cdt_num_of_valid_dat))
        self.test_gold_rels = test_gold_rels
        self.all_null_rels = all_null_rels

        self.test_cdt_num_of_rels = [len(cdts) for cdts in cdt_rels_of_q]
        self.test_gold_num_of_rels = [len(cdts) for cdts in test_gold_rels]
        return smps_qq, smps_qr, cdt_num_of_valid_dat, gold_rel_num_q, groups_of_rels_cnt, cdt_rels_of_q

    def a_f2x(self, f):
        gold_rel, cdt_rels, qs = self.parse_f(f, silent=True)
        smps_qq = []
        smps_rr = []
        smps_qr = []
        smps_rq = []
        # relation 是只用后边的还是前面的也用了？ 先试一下只用后边的(后面指的是‘/’后面的部分)
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)
            q_c_emb = self.get_q_elmo_rep(q.split())
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                gr_x_emb = self.elmo_of_all_rels[gr]
                gr_x = self.idxs_of_rel(gr)
                for r in cdt_rs:
                    r_x_emb = self.elmo_of_all_rels[r]
                    if r in grs:
                        continue
                    # gr_q_idx = self.rep_of_question(self.question_of_the_rel_random(gr, tmp_question=q))
                    # r_q_idx = self.rep_of_question(self.question_of_the_rel_random(r))
                    r_x = self.idxs_of_rel(r)
                    smps_rr.append((q_x, gr_x, r_x, q_c_emb, gr_x_emb, r_x_emb))
            if self.debug:
                if len(smps_rr) > 500:
                    break
                    # if sum(gr_q_idx) > 0:
                    #     smps_qr.append((q_x, gr_q_idx, r_x))
                    #     if sum(r_q_idx) > 0:
                    #         smps_qq.append((q_x, gr_q_idx, r_q_idx))
                    # if sum(r_q_idx) > 0:
                    #     smps_rq.append((q_x, gr_x, r_q_idx))
        return smps_rr, smps_rq, smps_qr, smps_qq

    # ############################################ padding ############################
    def dynamic_padding(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            if hasattr(idxs[0], 'shape'):
                to_add = np.zeros((max_len - len(idxs), idxs.shape[1]))
                idxs_batch_padded.append(np.concatenate([idxs, to_add], axis=0))
            else:
                to_add = [0] * (max_len - len(idxs))
                idxs_batch_padded.append(idxs + to_add)

        return idxs_batch_padded

    def dynamic_padding_train_batch(self, batch):
        """
        g_rel_alias_q_idxs 的维度应该是 batch * question_num * question_len  需要padding的是最后一个维度
        :param batch:
        :return:
        """

        q_idxs, g_alias_q_idxs, neg_alias_q_idxs, q_embs, g_embs, neg_embs= batch
        q_idxs = self.dynamic_padding(q_idxs)
        g_alias_q_idxs = self.dynamic_padding(g_alias_q_idxs)
        neg_alias_q_idxs = self.dynamic_padding(neg_alias_q_idxs)

        q_embs = self.dynamic_padding(q_embs)
        g_embs = self.dynamic_padding(g_embs)
        neg_embs = self.dynamic_padding(neg_embs)

        batch = q_idxs, g_alias_q_idxs, neg_alias_q_idxs, q_embs, g_embs, neg_embs
        return list(np.array(i) for i in batch)

    def max_alias_len(self, g_rel_alias_q_idxs):
        vs = []
        for q_idxs in g_rel_alias_q_idxs:
            vs.append(max([len(q) for q in q_idxs]))
        return max(vs)

    def dynamic_padding_valid_batch(self, batch):
        q_idxs, cdt_rel_idxs, flags, qembs, rembs = batch
        q_idxs = self.dynamic_padding(q_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)
        qembs = self.dynamic_padding(qembs)
        rembs = self.dynamic_padding(rembs)
        batch = q_idxs, cdt_rel_idxs, flags, qembs, rembs
        return list(np.array(i) for i in batch)

    # ############################################ batches ############################
    # def get_train_batchs(self):
    #     # self.logger.info('len(self.smps_rr, self.smps_rq, self.smps_qr, self.smps_qq)')
    #     # self.logger.info(list(len(pairs) for pairs in (self.smps_rr, self.smps_rq, self.smps_qr, self.smps_qq)))
    #     # return (self.get_train_batchs_per(pairs) for pairs in (self.smps_rr, self.smps_rq, self.smps_qr, self.smps_qq))
    #     return (self.get_train_batchs_per(pairs) for pairs in (self.smps_rr, ))[0]

    def get_train_batchs(self):
        train_smp_pairs = self.smps_rr
        train_smp_pair_num = len(train_smp_pairs)
        indices = list(range(train_smp_pair_num))
        random.shuffle(indices)
        batch = [[] for i in range(6)]
        idx = 0
        while idx < len(indices):
            items = train_smp_pairs[indices[idx]]
            for i, item in enumerate(items):
                batch[i].append(item)
            if len(batch[0]) == self.opt.train_batchsize or idx + 1 == train_smp_pair_num:
                batch = self.dynamic_padding_train_batch(batch)
                yield batch
                batch = [[] for i in range(6)]
            idx += 1

    def valid_or_test_batches(self, which='qr'):
        the_smps = self.test_smps_qr if which == 'qr' else self.test_smp_pairs_qq
        smp_pair_num = len(the_smps)
        batch = [[] for i in range(5)]
        for idx, smp in enumerate(the_smps):
            for i, item in enumerate(smp):
                batch[i].append(item)
            if len(batch[0]) == self.opt.valid_batchsize or idx + 1 == smp_pair_num:
                batch = self.dynamic_padding_valid_batch(batch)

                yield batch
                batch = [[] for i in range(5)]

    @staticmethod
    def parse_f(fnm, silent=False):
        """
        解析simpleQuestion数据集
        :param fnm:
        :return:
        """
        lns = [ln.strip('\n') for ln in open(fnm).readlines()]
        gold_rel = []
        cdt_rels = []
        qs = []
        if not silent:
            print(len(lns), fnm)
        for ln in lns:
            gr, cdt_rs, q = ln.split('\t')
            qs.append(q)
            if len(gr.split()) > 1:
                multi_gold_r = []
                for r in gr.split():
                    multi_gold_r.append(int(r) - 1)  # 这里写错啦   简直不明白为什么还会有50%的正确率   后边不都是错误了吗？
                gold_rel.append(multi_gold_r)
            else:
                gold_rel.append(int(gr) - 1)

            if cdt_rs == 'noNegativeAnswer':
                cdt_rels.append([])
            else:
                cdt_rels.append([int(i) - 1 for i in cdt_rs.split()])
        return gold_rel, cdt_rels, qs


if __name__ == '__main__':
    from logging import getLogger
    from RDOfWebQA.WebQAParapmeters import RDParameters
    WebQADataManager(RDParameters(0), getLogger('t'))
    pass
    # 怎么从中抽取关键词也是一个问题啊

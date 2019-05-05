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

import word_vocab
import dat.file_pathes as dt_p


class SimpleQADataManager:
    def __init__(self, opt, logger):
        self.opt = opt
        self.alias_q_max_len = self.opt.alias_q_max_len

        self.vocab = word_vocab.WordVocab(dt_p.simple_qa_vocab_f, emb_dim=300)

        q_fs = [u'{}{}.replace_ne.withpool'.format(dt_p.simple_qa_dp, i) for i in 'valid train test'.split()]
        self.q_fs = q_fs
        self.rels = [ln.strip('\n') for ln in open(dt_p.simple_qa_rel_f).readlines()]
        self.rel_questions = self.get_query_of_rel()
        self.rel_wh_word = self.get_query_wh_of_rel()
        self.rel_wh_word_sts = self.get_query_wh_of_rel_more()
        self.rel_alias_words = {}
        self.get_train = False
        self.train_batch_indices_cache = None
        self.train_pair_like_i_1 = {}  # 上一个epoch的结果
        self.train_pair_like_i = {}
        self.parsed_train_dat_cache = self.parse_f(q_fs[1])

        self.valid_smp_pairs, self.cdt_num_of_valid_dat = self.a_f2x_valid_and_test(q_fs[0])
        self.test_smp_pairs, self.cdt_num_of_test_dat = self.a_f2x_valid_and_test(q_fs[2])
        self.get_train = True
        self.train_smp_pairs = self.a_f2x(self.q_fs[1])
        self.get_train = False
        # logger.info('train_smp_pair_num={}， valid_smp_pair_num={}, test_smp_pair_num={}'
        #             ''.format(len(self.train_smp_pairs),
        #                       len(self.valid_smp_pairs),
        #                       len(self.test_smp_pairs)))

    def get_query_wh_of_rel(self):
        rel_questions = defaultdict(list)

        # for rel_idx, rel in enumerate(self.rels):
        #     rel_questions[rel_idx].append(u' '.join(self.words_of_web_rel(rel_idx)))
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[1])
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        rel_questions_wh = dict()
        # whs = 'when where who what'.split()
        whs = 'when where who what'.split()

        for r, qs in rel_questions.items():
            whs_cnt = defaultdict(int)
            for q in qs:
                for wh in whs:
                    if wh in set(q.split()):
                        whs_cnt[wh] += 1
            max_wh = max(whs, key=lambda wh: whs_cnt[wh])
            rel_questions_wh[r] = max_wh  # 这里可以大有作为吧?
            rel_name = self.rels[r]

        return rel_questions_wh

    def get_query_wh_of_rel_more(self):
        rel_questions = defaultdict(list)

        # for rel_idx, rel in enumerate(self.rels):
        #     rel_questions[rel_idx].append(u' '.join(self.words_of_web_rel(rel_idx)))
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[1])
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        rel_questions_wh_sts = dict()
        # whs = 'when where who what'# country track tracks song band artist film movie year month'.split() # + which
        whs = 'when where who what'.split()
        self.whs_idxs = self.vocab.seqword2id(whs)
        for r, qs in rel_questions.items():
            whs_cnt = [0] * len(whs)
            for q in qs:
                for wh_idx, wh in enumerate(whs):
                    if wh in set(q.split()):
                        whs_cnt[wh_idx] += 1
            whs_cnt = np.asarray(whs_cnt) / len(qs) # sum(whs_cnt) + 0.00001) # len(qs)  #
            rel_questions_wh_sts[r] = whs_cnt  # 这里可以大有作为吧?
            rel_name = self.rels[r]

        return rel_questions_wh_sts

    def get_query_of_rel(self):
        rel_questions = defaultdict(list)
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[1])
        for gr, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            rel_questions[gr].append(q)

        rel_q_cnt = {rel: len(qs) for rel, qs in rel_questions.items()}
        print(sum([cnt for rel, cnt in rel_q_cnt.items()]))
        print(sum([cnt for rel, cnt in rel_q_cnt.items() if cnt < 10]))
        print(sum([cnt for rel, cnt in rel_q_cnt.items() if cnt < 5]))
        print(sum([cnt for rel, cnt in rel_q_cnt.items() if cnt >= 10]))
        return rel_questions

    # ######################################### trans to idx ################################################
    @staticmethod
    def words_of_rel(rel):
        """
        多个地方会用到，万一要改比较方便
        """
        # return u'_'.join([rel.split('/')[-2]]).split('_') + rel.split('/')[-1].split('_')
        # return u'_'.join([rel.split('/')[-2]]).split('_') + rel.split('/')[-1].split('_') # 这个版本好像还行啊 验证集可以到94%

        return rel.split('/')[-1].split('_')

    def idxs_of_rel(self, rel_idx):
        """

        :param rel_idx: 在self.rels中的编号
        :return:
        """
        r_ws = self.words_of_rel(self.rels[rel_idx])
        # rel_x = [self.vocab.word2id(self.rels[rel_idx]), self.vocab.word2id(self.vocab.RELSEP)] + \
        # if self.get_train and random.randint(1, 100) > 50:
        #     rel_x = [self.vocab.word2id(self.rels[rel_idx])]
        # else:
        rel_x = self.vocab.seqword2id(r_ws) + [self.vocab.word2id(self.rels[rel_idx])]

        # rel_x = self.vocab.seqword2id(['']) + rel_x + self.vocab.seqword2id([''])
        if not self.opt.use_wh_hot:
            return rel_x

        wh = self.rel_wh_word.get(rel_idx, 'nu')   # 就搞这一点点,出了那么多的错误   ╮(╯▽╰)╭

        if self.get_train and random.randint(1, 100) > 10:
            wh = 'nu'
        rel_x = [self.vocab.word2id(wh)] + rel_x   # 卧槽 一直有这东西？
        return rel_x

    def question_of_the_rel(self, rel, tmp_question):
        cdts = self.rel_questions.get(rel, [None])
        question = random.choice(cdts) # + [None])
        while question == tmp_question:
            if len(set(cdts)) > 1:
                question = random.choice(cdts )# + [None])
            else:
                return ''
        if question is None:
            return ''
        return question

    def rep_of_alias_questions(self, rel, tmp_question, for_valid=False):
        if for_valid:
            rel_questions = self.rel_questions.get(rel, [])[:self.opt.alias_num]
            while len(rel_questions) < self.opt.alias_num:
                rel_questions.append('')
        else:
            rel_questions = [self.question_of_the_rel(rel, tmp_question) for i in range(self.opt.alias_num)]
        rep_of_questions = [self.rep_of_question(q) for q in rel_questions]
        return rep_of_questions

    def rep_of_question(self, q):
        return self.vocab.seqword2id(q.split())

    def a_f2x(self, f):
        gold_rel, cdt_rels, qs = self.parsed_train_dat_cache
        smps = []
        for gr, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)
            gr_x = self.idxs_of_rel(gr)
            # gr_alias_q_x = self.rep_of_alias_questions(gr, q)
            grx_wh_sts = self.rel_wh_word_sts.get(gr, [0] * len(self.whs_idxs))
            # if random.randint(0, 100) > 10:
            #     grx_wh_sts = [0] * len(self.whs_idxs)
            for r in cdt_rs:
                if r == gr:
                    continue
                r_x = self.idxs_of_rel(r)
                rx_wh_sts = self.rel_wh_word_sts.get(r, [0] * len(self.whs_idxs))
                # if random.randint(0, 100) > 10:
                #     rx_wh_sts = [0] * len(self.whs_idxs)

                # r_alias_q_x = self.rep_of_alias_questions(r, q)
                smps.append((q_x, gr_x, r_x, grx_wh_sts, rx_wh_sts))

        # print('q', max(map(len, q_xs)), 'g r', max(map(len, r_xs)), 'neg r', max(map(len, neg_r_xs)))
        # q 35 r 18
        return smps

    def a_f2x_valid_and_test(self, f):
        """
        :param f:
        :return:
        """
        gold_rel, cdt_rels, qs = self.parse_f(f)
        smps = []
        cdt_num_of_valid_dat = []
        # question_num = 0
        # candidate_num = 0
        for q_idx, (gr, cdt_rs, q) in tqdm(enumerate(zip(gold_rel, cdt_rels, qs))):
            q_x = self.rep_of_question(q)
            q_cdt_rels = [gr] + [r for r in set(cdt_rs) if r != gr]

            cdt_num_of_valid_dat.append(len(q_cdt_rels))
            # question_num += 1
            # candidate_num += len(q_cdt_rels)
            for r_idx in q_cdt_rels:
                rx1 = self.idxs_of_rel(r_idx)
                rx_wh_sts = self.rel_wh_word_sts.get(r_idx, [0] * len(self.whs_idxs))
                smps.append((q_x, rx1, rx_wh_sts))
            #     r_alias_q_x = self.rep_of_alias_questions(r_idx, q, for_valid=True)
            #     smps.append((q_x, rx1, r_alias_q_x))
        return smps, cdt_num_of_valid_dat

    # ############################################ padding ############################
    def dynamic_padding(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            to_add = [0] * (max_len - len(idxs))
            idxs_batch_padded.append(idxs + to_add)
        return idxs_batch_padded

    def dynamic_padding_train_batch(self, batch):
        """
        g_rel_alias_q_idxs 的维度应该是 batch * question_num * question_len  需要padding的是最后一个维度
        :param batch:
        :return:
        """
        # q_idxs, g_rel_idxs, g_rel_alias_q_idxs, cdt_rel_idxs, cdt_rel_alias_q_idxs = batch
        q_idxs, g_rel_idxs, cdt_rel_idxs, gwh_sts, negwh_sts = batch
        q_idxs = self.dynamic_padding(q_idxs)
        g_rel_idxs = self.dynamic_padding(g_rel_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        # g_rel_alias_q_idxs = [self.dynamic_padding(q_idxs, self.alias_q_max_len) for q_idxs in g_rel_alias_q_idxs]
        # cdt_rel_alias_q_idxs = [self.dynamic_padding(q_idxs, self.alias_q_max_len) for q_idxs in cdt_rel_alias_q_idxs]
        # batch = q_idxs, g_rel_idxs, g_rel_alias_q_idxs, cdt_rel_idxs, cdt_rel_alias_q_idxs  # 这个alias 可是不得了。不能用的。
        batch = q_idxs, g_rel_idxs, cdt_rel_idxs, gwh_sts, negwh_sts # 这个alias 可是不得了。不能用的。

        return list(np.array(i) for i in batch)

    def dynamic_padding_valid_batch(self, batch):
        q_idxs, cdt_rel_idxs, wh_sts, _  = batch
        q_idxs = self.dynamic_padding(q_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)

        # cdt_rel_alias_q_idxs = [self.dynamic_padding(q_idxs, self.alias_q_max_len) for q_idxs in cdt_rel_alias_q_idxs]
        batch = q_idxs, cdt_rel_idxs, wh_sts,  _ # , cdt_rel_alias_q_idxs
        return list(np.array(i) for i in batch)

    # ############################################ batches ############################
    def get_train_batchs(self, epoch, for_keras=False):
        self.train_smp_pair_num = len(self.train_smp_pairs)
        print(self.train_smp_pair_num, 'training sample num')

        indices = list(range(self.train_smp_pair_num))
        # if epoch < 1:
        random.shuffle(indices)
        batch = [[] for i in range(5)]
        idx = 0
        batch_indices = []

        # # 到时候在这里写一个 select indices 然后进行选择
        if epoch > 1 and self.opt.with_selection:
            indices = self.select_smps(epoch)
        # random.shuffle(indices)

        self.train_pair_like_i_1 = {k: v for k, v in self.train_pair_like_i.items()}

        while idx < len(indices):
            items = self.train_smp_pairs[indices[idx]]
            batch_indices.append(idx)
            for i, item in enumerate(items):
                batch[i].append(item)
            if len(batch[0]) == self.opt.train_batchsize or idx + 1 == self.train_smp_pair_num:
                batch = self.dynamic_padding_train_batch(batch)
                self.train_batch_indices_cache = batch_indices
                if for_keras:
                    batch[2] = batch[2].reshape(batch[2].shape[0], batch[2].shape[1] * batch[2].shape[2])
                    batch[4] = batch[4].reshape(batch[4].shape[0], batch[4].shape[1] * batch[4].shape[2])
                yield batch
                batch = [[] for i in range(5)]
                batch_indices = []
            idx += 1

    def select_smps(self, epoch):
        indices = list(range(len(self.train_smp_pairs)))
        diffs = []  # 在两轮迭代中的差值
        last_indices = []
        need_review_indices = []
        for idx in indices:
            if idx not in self.train_pair_like_i:
                self.train_pair_like_i[idx] = self.train_pair_like_i_1[idx]
            if self.train_pair_like_i[idx] > 0:
                last_indices.append(idx)
                cost_i = self.train_pair_like_i[idx]
                diffs.append(cost_i)
            else:
                need_review_indices.append(idx)

        min_dif = min(diffs)
        max_dif = max(diffs)
        diffs = (np.array(diffs) - min_dif) / (max_dif - min_dif)

        # indices_selected = pd.Series(last_indices).sample(frac=0.8, weights=diffs, replace=False)
        indices_selected = last_indices
        indices_selected = list(indices_selected)

        review_indices = random.sample(need_review_indices, k=int(self.opt.review_rate * len(need_review_indices)))
        # review_indices = random.sample(need_review_indices, k=int(0.3 * len(need_review_indices)))

        assert len(indices_selected) == len(set(indices_selected))

        if epoch > 10:
            # random.shuffle(indices_selected)
            random.shuffle(review_indices)
            final_indices = indices_selected + review_indices
        else:
            final_indices = indices
            random.shuffle(final_indices)
        # remove_dup_idxs

        # return sorted(indices, key=lambda idx: self.train_pair_like_i[idx], reverse=True)
        print(f'selected indices={len(indices_selected)}, review indices={len(review_indices)}')

        return final_indices

    def valid_or_test_batches(self, valid_or_test='valid', for_keras=False):
        the_smps = self.valid_smp_pairs if valid_or_test == 'valid' else self.test_smp_pairs
        smp_pair_num = len(the_smps)
        batch = [[] for i in range(4)]
        for idx, smp in enumerate(the_smps):
            for i, item in enumerate(smp):
                batch[i].append(item)
            if len(batch[0]) == self.opt.valid_batchsize or idx + 1 == smp_pair_num:
                batch = self.dynamic_padding_valid_batch(batch)
                if for_keras:
                    batch[2] = batch[2].reshape(batch[2].shape[0], batch[2].shape[1] * batch[2].shape[2])
                yield batch
                batch = [[] for i in range(4)]

    def record_pair_values(self, batch_score):
        for idx, s in zip(self.train_batch_indices_cache, batch_score):
            self.train_pair_like_i[idx] = s

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
    pass

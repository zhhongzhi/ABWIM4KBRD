# -*- encoding: utf-8 -*-
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


class WebQADataManager:
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger
        self.vocab = word_vocab.WordVocab(dt_p.webqa_vocab_f, emb_dim=300)
        q_fs = [u'{}WebQSP.RE.{}.with_boundary.withpool.dlnlp.txt'.format(dt_p.web_q_f_dp, i) for i in 'train test'.split()]

        self.q_fs = q_fs
        self.rels = [ln.strip('\n') for ln in open(dt_p.web_qa_rel_f).readlines()]
        self.rel_alias_words = {}
        self.get_train = False
        self.train_batch_indices_cache = None
        self.train_pair_like_i_1 = {}  # 上一个epoch的结果
        self.train_pair_like_i = {}

        self.rel_questions = self.get_query_of_rel()
        self.rel_wh_word = self.get_query_wh_of_rel()
        self.rel_wh_word_sts = self.get_query_wh_of_rel_more()

        self.test_smps_qr, self.cdt_num_of_test_dat, self.gold_rel_num_test_dat, self.groups_of_rels_cnt,\
            self.cdt_rels_of_q = self.a_f2x_valid_and_test(q_fs[1])
        self.smps_rr = self.a_f2x(self.q_fs[0])
        self.ever_contribute = set()

    def get_query_wh_of_rel(self):
        rel_questions = defaultdict(list)

        # for rel_idx, rel in enumerate(self.rels):
        #     rel_questions[rel_idx].append(u' '.join(self.words_of_web_rel(rel_idx)))
        gold_rel, cdt_rels, qs = self.parse_f(self.q_fs[0])
        for grs, cdt_rs, q in tqdm(zip(gold_rel, cdt_rels, qs)):
            if not isinstance(grs, list):
                grs = [grs]
            for gr in grs:
                rel_questions[gr].append(q)  # u' '.join(q.split()[1:-1]))
        rel_questions_wh = dict()
        whs = 'when where who what'.split()
        for r, qs in rel_questions.items():
            whs_cnt = defaultdict(int)
            for q in qs:
                for wh in whs:
                    if wh in set(q.split()):
                        whs_cnt[wh] += 1
            max_wh = max(whs, key=lambda wh: whs_cnt[wh])
            rel_questions_wh[r] = max_wh  # 这里可以大有作为吧?

        # whs = 'when where who what'.split()
        self.whs_idxs = self.vocab.seqword2id(whs)

        return rel_questions_wh
        # rel_questions_sorted = {}

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
        if not self.opt.use_wh_hot:
            return ws + rels
        wh = self.rel_wh_word.get(rel_idx, 'm')   # 就搞这一点点,出了那么多的错误   ╮(╯▽╰)╭
        if self.get_train and random.randint(1, 100) > 10:
            return ['m'] + ws + rels
            # return ws + rels
        # if wh is 'none':
        #     return ws + rels
        return [wh] + ws + rels

    def words_of_a_rel(self, rel):
        """
        多个地方会用到，万一要改比较方便
        """
        return rel.split('/')[-1].split('_')

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
        whs = 'when where who what'.split()# country track tracks song band artist film movie year month'.split() # + which
        # whs = 'when where who what'.split()
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
            groups = []
            g_rels_q_num = 0
            cdt_q_num = 0
            tmp_cdt_rels = []
            assert len(q_cdt_rels) == len(flags)
            null_rels = []
            for r, flag in zip(q_cdt_rels, flags):
                r_x_wh = self.rel_wh_word_sts.get(r, [0] * len(self.whs_idxs))

                smps_qr.append((q_x, self.idxs_of_rel(r), 1 if flag == 'g' else 0, r_x_wh))
                questions = self.rel_questions.get(r, [])[:self.opt.valid_q_max_num]
                if len(questions) == 0:
                    null_rels.append(r)
                if flag == 'g':
                    g_rels_q_num += len(questions)
                cdt_q_num += len(questions)
                groups.append(len(questions))
                tmp_cdt_rels.append(r)
            cdt_rels_of_q.append(tmp_cdt_rels)
            groups_of_rels_cnt.append(groups)
            cdt_num_of_valid_dat.append(cdt_q_num)
            gold_rel_num_q.append(g_rels_q_num)
            all_null_rels.append(null_rels)
        self.test_gold_rels = test_gold_rels
        self.all_null_rels = all_null_rels

        self.test_cdt_num_of_rels = [len(cdts) for cdts in cdt_rels_of_q]
        self.test_gold_num_of_rels = [len(cdts) for cdts in test_gold_rels]
        return smps_qr, cdt_num_of_valid_dat, gold_rel_num_q, groups_of_rels_cnt, cdt_rels_of_q

    def a_f2x(self, f):
        gold_rel, cdt_rels, qs = self.parse_f(f, silent=True)
        smps_rr = []
        self.get_train = True
        gt_1 = 0

        more_than_one_golds = {}
        # relation 是只用后边的还是前面的也用了？ 先试一下只用后边的(后面指的是‘/’后面的部分)
        for q_idx, (grs, cdt_rs, q) in enumerate(zip(gold_rel, cdt_rels, qs)):
            q_x = self.rep_of_question(q)
            if not isinstance(grs, list):
                grs = [grs]
            dup_q = defaultdict(list)
            for gr in grs:
                gr_x = self.idxs_of_rel(gr)
                gr_x_wh = self.rel_wh_word_sts.get(gr, [0] * len(self.whs_idxs))
                for r in cdt_rs:
                    if r in grs:
                        continue
                    smps_rr_idx = len(smps_rr)
                    dup_q[gr].append(smps_rr_idx)
                    r_x = self.idxs_of_rel(r)
                    r_x_wh = self.rel_wh_word_sts.get(r, [0] * len(self.whs_idxs))
                    smps_rr.append((q_x, gr_x, r_x, gr_x_wh, r_x_wh))
            if len(grs) > 1:
                gt_1 += 1
            more_than_one_golds[q_idx] = dup_q
        self.get_train = False
        self.more_than_one_golds = more_than_one_golds
        return smps_rr

    # ############################################ padding ############################
    def dynamic_padding(self, idxs_batch, max_len=None):
        if max_len is None:
            max_len = max([len(idxs) for idxs in idxs_batch])
        idxs_batch_padded = []
        for idxs in idxs_batch:
            idxs = idxs[:max_len]
            to_add = [0] * (max_len - len(idxs))
            # idxs_batch_padded.append(idxs + to_add)
            idxs_batch_padded.append(to_add + idxs)
        return idxs_batch_padded

    def dynamic_padding_train_batch(self, batch):
        """
        g_rel_alias_q_idxs 的维度应该是 batch * question_num * question_len  需要padding的是最后一个维度
        :param batch:
        :return:
        """

        q_idxs, g_alias_q_idxs, neg_alias_q_idxs, g_wh_xs, neg_wh_xs = batch
        q_idxs = self.dynamic_padding(q_idxs)
        g_alias_q_idxs = self.dynamic_padding(g_alias_q_idxs)
        neg_alias_q_idxs = self.dynamic_padding(neg_alias_q_idxs)

        batch = q_idxs, g_alias_q_idxs, neg_alias_q_idxs, g_wh_xs, neg_wh_xs
        return list(np.array(i) for i in batch)

    def max_alias_len(self, g_rel_alias_q_idxs):
        vs = []
        for q_idxs in g_rel_alias_q_idxs:
            vs.append(max([len(q) for q in q_idxs]))
        return max(vs)

    def dynamic_padding_valid_batch(self, batch):
        q_idxs, cdt_rel_idxs, flags, wh_xs = batch
        q_idxs = self.dynamic_padding(q_idxs)
        cdt_rel_idxs = self.dynamic_padding(cdt_rel_idxs)
        batch = q_idxs, cdt_rel_idxs, flags, wh_xs
        return list(np.array(i) for i in batch)

    def get_train_batchs(self, epoch):
        self.smps_rr = self.a_f2x(self.q_fs[0])
        train_smp_pairs = self.smps_rr
        train_smp_pair_num = len(train_smp_pairs)
        indices = list(range(train_smp_pair_num))

        random.shuffle(indices)
        # # 到时候在这里写一个 select indices 然后进行选择
        if epoch > 1 and self.opt.with_selection:
            indices = self.select_smps(epoch)
        self.train_pair_like_i_1 = {k: v for k, v in self.train_pair_like_i.items()}

        batch = [[] for i in range(5)]
        batch_indices = []
        idx = 0
        while idx < len(indices):
            items = train_smp_pairs[indices[idx]]
            for i, item in enumerate(items):
                batch[i].append(item)
            batch_indices.append(indices[idx])
            if len(batch[0]) == self.opt.train_batchsize or idx + 1 == train_smp_pair_num:
                batch = self.dynamic_padding_train_batch(batch)
                self.train_batch_indices_cache = batch_indices
                yield batch
                batch = [[] for i in range(5)]
                batch_indices = []
            idx += 1

    def record_pair_values(self, batch_score):
        for idx, s in zip(self.train_batch_indices_cache, batch_score):
            self.train_pair_like_i[idx] = s

    def select_smps(self, epoch):
        remove_dup_idxs = set()
        indices = list(range(len(self.smps_rr)))
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
            random.shuffle(indices_selected)
            random.shuffle(review_indices)
            final_indices = [i for i in indices_selected + review_indices if i not in remove_dup_idxs]
        else:
            final_indices = [i for i in indices if i not in remove_dup_idxs]
            random.shuffle(final_indices)
        # remove_dup_idxs
        print(f'selected indices={len(indices_selected)}, review indices={len(review_indices)}')

        print(f'remove item {len([i for i in indices_selected if i in remove_dup_idxs])}')
        return final_indices

    def valid_or_test_batches(self, which='qr'):
        the_smps = self.test_smps_qr if which == 'qr' else self.test_smp_pairs_qq
        smp_pair_num = len(the_smps)
        batch = [[] for i in range(4)]
        for idx, smp in enumerate(the_smps):
            for i, item in enumerate(smp):
                batch[i].append(item)
            if len(batch[0]) == self.opt.valid_batchsize or idx + 1 == smp_pair_num:
                batch = self.dynamic_padding_valid_batch(batch)
                yield batch
                batch = [[] for i in range(4)]

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

# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""
import sys
import logging
import json
import os
import math
import time

sys.path.append('/home/hongzhi/wp/KBQA_nlpcc_2018/')
dev = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(dev)

import numpy as np
import torch

from ABWIMWebQSP.JointModel import TraditionalRDModel as JointModel
from ABWIMWebQSP.Joint_webqa_paras import JointWebqaParameters
from ABWIMWebQSP.WebQADataManager import WebQADataManager
from utils import AverageMeter


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class NerTrain:
    def __init__(self, opt=JointWebqaParameters(dev)):
        self.opt = opt
        model_dir = self.opt.model_dir
        self.loger = self.setup_loger()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.dt = WebQADataManager(self.opt, self.loger)
        # self.train()
        if self.opt.resume_dsrc_flag:
            self.model = self.resume()
        else:
            self.model = JointModel(opt=self.opt, emb_vs=self.dt.vocab.vs)
        self.model.cuda()
        self.loger.info(json.dumps(dict(self.opt), indent=True))
        self.best_valid_f1 = 0
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.error_in_traindat = 0

    def train(self):
        epoches = 800
        for epoch in range(epoches):
            batches_rr = self.dt.get_train_batchs()
            num = 0

            for idx, batch_rr in enumerate(batches_rr):
                self.model.update(batch_rr)
                # self.model.update(batch_rr, 'rr')
                # self.model.adv_update(batch_rr, 'rr')
                # if batches_qq is not None:
                #     try:
                #         if idx % 5 == 0:
                #             # self.model.update(batches_qq.__next__(), 'qq')
                #             self.model.adv_update(batches_qq.__next__(), 'qq')
                #             pass
                #     except StopIteration:
                #         batches_qq = None
                #

                # if batches_rq is not None:
                #     try:
                #         self.model.update(batches_rq.__next__(), 'rq')
                #     except StopIteration:
                #         batches_rq = None
                #
                # if batches_qr is not None:
                #     try:
                #         self.model.update(batches_qr.__next__(), 'qr')
                #     except StopIteration:
                #         batches_qr = None

                if idx % 1000 == 0 and idx > 0:
                    self.loger.info('batch_idx={}, loss={}, adv_loss={}'.format(idx, self.model.train_loss.avg,
                                                                                self.model.adv_train_loss.avg))
                num += 1
            self.valid_it(epoch)
            self.model.zero_loss()
            self.train_loss = AverageMeter()
            self.error_in_traindat = 0

    def valid_it(self, epoch):
        self.loger.info('================================= epoch: {} =================================='.format(epoch))
        self.loger.info('loss={}'.format(self.model.train_loss.avg))
        for valid_or_test in ('valid', 'test'):
            if valid_or_test == 'valid':
                scores = self.predict_all_batches(self.dt.valid_or_test_batches('qr'), 'r')
                # scores_q_sim = self.predict_all_batches(self.dt.valid_or_test_batches('qq'), 'q')
                rel_acc = self.infer_acc_rel_v(scores, self.dt.test_cdt_num_of_rels, self.dt.test_gold_num_of_rels)

                # q_acc = self.infer_acc(scores_q_sim, self.dt.cdt_num_of_test_dat, self.dt.gold_rel_num_test_dat,
                #                     self.dt.groups_of_rels_cnt)
                #
                # joint_acc = self.joint_infer_acc(scores_q_sim, scores, self.dt.cdt_num_of_test_dat, self.dt.gold_rel_num_test_dat,
                #                     self.dt.groups_of_rels_cnt)
                #
                # joint_acc_1 = self.joint_infer_acc(scores_q_sim, scores, self.dt.cdt_num_of_test_dat, self.dt.gold_rel_num_test_dat,
                #                     self.dt.groups_of_rels_cnt, alpha=1)
                #
                # joint_acc_0 = self.joint_infer_acc(scores_q_sim, scores, self.dt.cdt_num_of_test_dat, self.dt.gold_rel_num_test_dat,
                #                     self.dt.groups_of_rels_cnt, alpha=0)
                # accs = []
                # for i in range(0, 10):
                #     i /= 10.0
                #     joint_acc_01 = self.joint_infer_acc(scores_q_sim, scores, self.dt.cdt_num_of_test_dat, self.dt.gold_rel_num_test_dat,
                #                         self.dt.groups_of_rels_cnt, alpha=i)
                #     accs.append((i, joint_acc_01))
                # self.loger.info(accs)

                if rel_acc > self.best_valid_f1:
                    self.loger.info('new best valid f1 found')
                    self.best_valid_f1 = rel_acc
                    # if joint_acc > 0.84:
                    #     fnm = self.opt.model_dir + 'model_epoch_{}.h5.'.format(epoch)
                    #     self.model.save(fnm, epoch)
                self.loger.info(u'r_f1={}, q_f1={}, joint_f1={}'.format(rel_acc, 0, 0))
                self.loger.info(u'joint_f1, alpha=1, v={}, alpha=0, v={}'.format(0, 0))

    def predict_all_batches(self, batches, which):
        all_scores = []
        for batch in batches:
            scores = self.model.predict_score_of_batch(batch)
            all_scores.append(scores)
        all_scores = np.concatenate(all_scores, axis=0)

        return [x for x in list(all_scores.reshape(-1))]

    def infer_acc_rel_v(self, scores, cdt_num_of_valid_dat, gold_res_num_of_valid_dat):
        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        acc_num = 0
        pos = 0
        for num, g_num in zip(cdt_num_of_valid_dat, gold_res_num_of_valid_dat):
            if num == 1:
                acc_num += 1
                pos += num
                continue

            if num == g_num:
                acc_num += 1
                pos += num
                continue

            q_scores = scores[pos: pos + num]
            # print(q_scores)
            if max(q_scores[:g_num]) > max(q_scores[g_num:]):
                acc_num += 1
            pos += num
        assert pos == len(scores)
        # print(acc_num, len(cdt_num_of_valid_dat))
        return acc_num / len(cdt_num_of_valid_dat)

    def infer_acc(self, scores, cdt_num_of_valid_dat, gold_res_num_of_valid_dat, groups_of_rels_cnt):
        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        acc_num = 0
        pos = 0
        for num, g_num, groups, rels, g_rels in zip(cdt_num_of_valid_dat, gold_res_num_of_valid_dat, groups_of_rels_cnt,
                                                    self.dt.cdt_rels_of_q,
                                                    self.dt.test_gold_rels):
            if num == 1:
                acc_num += 1
                pos += num
                continue

            if num == g_num:
                acc_num += 1
                pos += num
                continue

            q_scores = scores[pos: pos + num]

            assert len(groups) == len(rels)
            gp_pos = 0
            per_rel_scores = {}
            assert len(groups) == len(rels)
            for n, the_rel in zip(groups, rels):
                group_scores = q_scores[gp_pos: gp_pos + n]
                per_rel_scores[the_rel] = mean(group_scores)  # 没有questions rel 这里的结果就是0了
                gp_pos += n
            assert gp_pos == len(q_scores)

            sorted_rels = sorted(per_rel_scores.keys(), key=lambda k: per_rel_scores[k])
            if sorted_rels[-1] in g_rels:
                acc_num += 1

            pos += num
        assert pos == len(scores)
        return acc_num / len(cdt_num_of_valid_dat)

    def joint_infer_acc(self, scores, all_rel_scores, cdt_num_of_valid_dat, gold_res_num_of_valid_dat, groups_of_rels_cnt,
                        alpha=0.5):
        assert sum(self.dt.test_cdt_num_of_rels) == len(all_rel_scores), \
            (sum(self.dt.test_cdt_num_of_rels), len(all_rel_scores))
        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        assert len(set(len(i) for i in (
                self.dt.test_cdt_num_of_rels,
                self.dt.test_gold_num_of_rels,
                cdt_num_of_valid_dat,
                gold_res_num_of_valid_dat,
                groups_of_rels_cnt,
                self.dt.cdt_rels_of_q,
                self.dt.test_gold_rels))) == 1

        acc_num = 0
        pos = 0
        pos_rel = 0
        for rel_cdt_num, rel_g_num, num, g_num, groups, rels, g_rels in zip(
                self.dt.test_cdt_num_of_rels,
                self.dt.test_gold_num_of_rels,
                cdt_num_of_valid_dat,
                gold_res_num_of_valid_dat,
                groups_of_rels_cnt,
                self.dt.cdt_rels_of_q,
                self.dt.test_gold_rels):

            if rel_cdt_num == 1:
                acc_num += 1
                pos += num
                pos_rel += rel_cdt_num
                continue

            if rel_cdt_num == rel_g_num:
                acc_num += 1
                pos += num
                pos_rel += rel_cdt_num
                continue

            q_scores = scores[pos: pos + num]
            rel_scores = all_rel_scores[pos_rel: pos_rel + rel_cdt_num]
            assert len(groups) == len(rels)
            assert len(rel_scores) == len(rels)
            gp_pos = 0
            per_rel_scores = {}
            for n, the_rel, rel_score in zip(groups, rels, rel_scores):
                group_scores = q_scores[gp_pos: gp_pos + n]
                # per_rel_scores[the_rel] = (1.0-alpha) * mean(group_scores) + alpha * rel_score
                # #
                if len(group_scores) > 0:
                    per_rel_scores[the_rel] = (1.0-alpha) * mean(group_scores) + alpha * rel_score
                else:
                    per_rel_scores[the_rel] = rel_score  # 没有questions rel 这里的结果就是0了
                    # per_rel_scores[the_rel] = (1.0-alpha) * mean(group_scores) + alpha * rel_score
                gp_pos += n
            assert gp_pos == len(q_scores)

            sorted_rels = sorted(per_rel_scores.keys(), key=lambda k: per_rel_scores[k])
            if sorted_rels[-1] in g_rels:
                acc_num += 1

            pos += num
            pos_rel += rel_cdt_num
        assert pos == len(scores)
        assert pos_rel == len(all_rel_scores), (pos_rel, len(all_rel_scores))
        return int(acc_num / len(cdt_num_of_valid_dat) * 100000) / 100000

    def infer_rel_scores(self):
        scores = self.predict_all_batches(self.dt.valid_or_test_batches(), 'r')
        print(self.infer_acc(scores, self.dt.cdt_num_of_test_dat, self.dt.gold_rel_num_test_dat,
                             self.dt.groups_of_rels_cnt))

        cdt_num_of_valid_dat = self.dt.cdt_num_of_test_dat
        gold_res_num_of_valid_dat = self.dt.gold_rel_num_test_dat
        groups_of_rels_cnt = self.dt.groups_of_rels_cnt

        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        acc_num = 0
        pos = 0

        all_rel_scores = []
        for num, g_num, groups, rels in zip(cdt_num_of_valid_dat, gold_res_num_of_valid_dat, groups_of_rels_cnt,
                                            self.dt.cdt_rels_of_q):
            per_rel_scores = {}
            if num == 1:
                acc_num += 1
                pos += num
                per_rel_scores[rels[0]] = scores[pos: pos + num]
                all_rel_scores.append(per_rel_scores)
                continue

            # if num == g_num:
            #     acc_num += 1
            #     pos += num
            #     per_rel_scores[rels[0]] = scores[pos: pos + num]   # 这里简单把两个score和在一起？ no  这种应该也有，好好弄
            #
            #     continue

            q_scores = scores[pos: pos + num]

            gp_pos = 0
            assert len(groups) == len(rels)
            for n, the_rel in zip(groups, rels):
                group_scores = q_scores[gp_pos: gp_pos + n]
                per_rel_scores[the_rel] = group_scores
                gp_pos += n
            assert gp_pos == len(q_scores)
            pos += num
            all_rel_scores.append(per_rel_scores)
        assert pos == len(scores)
        return all_rel_scores, None

    def setup_loger(self):
        # setup logger
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.opt.log_file)
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        log.addHandler(fh)
        log.addHandler(ch)
        return log

    #
    def resume(self):
        self.loger.info('[loading previous model...]')
        checkpoint = torch.load(self.opt.trained_model)
        opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        self.loger.info(json.dumps(dict(opt), indent=True))
        model = JointModel(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict)  # 重启的时候
        # model.embedding_ly.weight.requires_grad = False
        return model


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


if __name__ == '__main__':
    opt = JointWebqaParameters(dev, train_idx='solely_rd_no_adv')

    # opt.q_hidden = 100
    t = NerTrain(opt=opt)
    # t.valid_it(-1)
    t.train()


# 结果好像没有以前好啊

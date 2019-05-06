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

dp = os.path.abspath(os.path.dirname(__file__))
dp = u'/'.join(os.path.split(dp)[:-1])
sys.path.append(dp)

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
            self.model = JointModel(opt=self.opt, emb_vs=self.dt.vocab.vs, wh_idxs=self.dt.whs_idxs)
        self.model.cuda()
        self.loger.info(json.dumps(dict(self.opt), indent=True))
        self.best_valid_f1 = 0
        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()
        self.error_in_traindat = 0

    def train(self):
        epoches = 800
        for epoch in range(epoches):
            batches_rr = self.dt.get_train_batchs(epoch)
            num = 0
            s = time.time()
            for idx, batch_rr in enumerate(batches_rr):
                batch_acc, scores = self.model.update(batch_rr)
                # batch_acc, scores = self.model.adv_update(batch_rr)
                self.dt.record_pair_values(scores)
                if idx % 1000 == 0 and idx > 0:
                    self.loger.info('batch_idx={}, loss={}, adv_loss={}'.format(idx, self.model.train_loss.avg,
                                                                                self.model.adv_train_loss.avg))
                num += 1
            e = time.time()
            print(e-s, 'training a epoch')

            self.valid_it(epoch)
            self.model.zero_loss()
            self.train_loss = AverageMeter()
            self.error_in_traindat = 0
            self.adjust_learning_rate(epoch)
            self.model.zero_loss()

    def adjust_learning_rate(self, epoch):
        optimizer = self.model.optimizer
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.5 ** int(epoch // 15))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def valid_it(self, epoch):
        self.loger.info('================================= epoch: {} =================================='.format(epoch))
        self.loger.info('loss={}'.format(self.model.train_loss.avg))

        self.loger.info(f'train_loss_not_null={self.model.train_loss_not_null.sum}, '
                        f'total_train_num={self.model.train_loss_not_null.count},')
        for valid_or_test in ('valid', 'test'):
            if valid_or_test == 'valid':
                scores = self.predict_all_batches(self.dt.valid_or_test_batches('qr'), 'r')
                # open('ABWIMNEW_scores.txt', 'w').writelines([u'{}\n'.format(s) for s in scores])

                # scores_q_sim = self.predict_all_batches(self.dt.valid_or_test_batches('qq'), 'q')
                rel_acc = self.infer_acc_rel_v(scores, self.dt.test_cdt_num_of_rels, self.dt.test_gold_num_of_rels)
                if rel_acc > self.best_valid_f1:
                    self.loger.info('new best valid f1 found')
                    self.best_valid_f1 = rel_acc
                    if rel_acc > 0.859:
                        fnm = self.opt.model_dir + 'model_epoch_{}.h5.'.format(-1)
                        self.model.save(fnm, epoch)
                self.loger.info(u'r_f1={}'.format(rel_acc))

    def predict_all_batches(self, batches, which):
        all_scores = []
        a = time.time()
        for batch in batches:
            scores = self.model.predict_score_of_batch(batch)
            all_scores.append(scores)
        all_scores = np.concatenate(all_scores, axis=0)
        b = time.time()
        print(b-a, 'inference time')

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
        model = JointModel(opt=opt, emb_vs=self.dt.vocab.vs, state_dict=state_dict, wh_idxs=self.dt.whs_idxs)  # 重启的时候
        # model.embedding_ly.weight.requires_grad = False
        return model


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


if __name__ == '__main__':
    opt = JointWebqaParameters(dev, train_idx='webqsp')

    opt.resume_dsrc_flag = False
    opt.trained_model = 'models/New_exp_idx_with_select_with_wh_0.2_kept_rate_adv_no_selection_on_1/model_epoch_-1.h5.'
    t = NerTrain(opt=opt)
    # t.valid_it(-1)
    t.train()

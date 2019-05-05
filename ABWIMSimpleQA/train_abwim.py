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

from ABWIMSimpleQA.JointModel import TraditionalRDModel as JointModel
from ABWIMSimpleQA.SimpleQAParapmeters import RDParameters as JointWebqaParameters
from ABWIMSimpleQA.SimpleQADataManager import SimpleQADataManager
from utils import AverageMeter


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class NerTrain:
    def __init__(self, opt=JointWebqaParameters(train_idx='none', dev=dev)):
        self.opt = opt
        model_dir = self.opt.model_dir
        self.loger = self.setup_loger()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.dt = SimpleQADataManager(self.opt, self.loger)
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
        batch_num = 0
        epoches = 100
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
                batch_num += 1
            e = time.time()
            # print(e-s, 'trining time')
            self.valid_it(epoch, batch_num)
            self.model.zero_loss()
            self.train_loss = AverageMeter()
            self.error_in_traindat = 0
            self.adjust_learning_rate(epoch)

    def adjust_learning_rate(self, epoch):
        optimizer = self.model.optimizer
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.opt.learning_rate * (0.75 ** int(epoch // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def valid_it(self, epoch, batch_num):
        self.loger.info('================================= epoch: {} =================================='.format(epoch))
        self.loger.info('loss={}'.format(self.model.train_loss.avg))

        self.loger.info(f'train_loss_not_null={self.model.train_loss_not_null.sum}, '
                        f'total_train_num={self.model.train_loss_not_null.count},')
        for valid_or_test in ('valid', 'test'):
                scores = self.predict_all_batches(self.dt.valid_or_test_batches(valid_or_test), 'r')
                rel_acc = self.infer_acc_rel_v(scores,
                                               self.dt.cdt_num_of_valid_dat if valid_or_test == 'valid' else self.dt.cdt_num_of_test_dat)

                if rel_acc > self.best_valid_f1 and valid_or_test == 'valid':
                    self.loger.info('new best valid f1 found')
                    self.best_valid_f1 = rel_acc
                    if rel_acc > 0.938:
                        fnm = self.opt.model_dir + 'model_epoch_{}.h5.'.format(epoch)
                        self.model.save(fnm, epoch)
                self.loger.info(u'{}, r_f1={}, batch_num={}, need'.format(valid_or_test, rel_acc, batch_num))

    def predict_all_batches(self, batches, which):
        s = time.time()
        all_scores = []
        for batch in batches:
            scores = self.model.predict_score_of_batch(batch)
            all_scores.append(scores)
        all_scores = np.concatenate(all_scores, axis=0)
        e = time.time()
        # print(e-s, 'inference time')

        return [x for x in list(all_scores.reshape(-1))]

    def infer_acc_rel_v(self, scores, cdt_num_of_valid_dat):
        assert sum(cdt_num_of_valid_dat) == len(scores), (sum(cdt_num_of_valid_dat), len(scores))
        acc_num = 0
        pos = 0
        for num in cdt_num_of_valid_dat:
            g_num = 1
            if num == 1:
                acc_num += 1
                pos += num
                continue

            if num == g_num:
                acc_num += 1
                pos += num
                continue

            q_scores = scores[pos: pos + num]
            # if len(q_scores) != len(set(q_scores)):
            #     print(q_scores, 'hey there is dup values')
            # print(q_scores)
            if max(q_scores[:g_num]) > max(q_scores[g_num:]):  # 而且这里用的是大于 很充沛了
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
    opt = JointWebqaParameters(dev=dev, train_idx='abwim')

    t = NerTrain(opt=opt)
    # t.valid_it(-1,0)
    t.train()


# 结果好像没有以前好啊

# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""

import logging
import sys

import torch
import torch.nn as nn
import layers
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
from utils import AverageMeter


class JointModel(nn.Module):
    def __init__(self, opt, emb_vs, padding_idx=0, state_dict=None):
        """
        就是一个seq label 的model,
        Embedding 层和RDModel share一个  不更新
        """
        super(JointModel, self).__init__()
        self.opt = opt
        self.logger = self.setup_loger()

        self.q_q_match = layers.SeqAttnMatchNoMask(opt.q_hidden * 2)
        match_in_dim = opt.q_hidden * 2 * (2 + 0)
        self.merge_sru_com = layers.StackedBRNNLSTM(
            input_size=match_in_dim,
            hidden_size=opt.merge_fts_dim,
            # num_layers=opt.sru_ly_num,
            num_layers=1,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers
        )

        self.o_liner_qq = nn.Linear(opt.merge_fts_dim * 2 * 2, 1, bias=True)

        parameters = [p for p in self.parameters() if p.requires_grad]

        if opt.optimizer == 'sgd':
            self.optimizer_qq = optim.SGD(parameters, opt.learning_rate,
                                       momentum=opt.momentum,
                                       weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamax':
            self.optimizer_qq = optim.Adamax(parameters, opt.learning_rate,
                                          weight_decay=opt.weight_decay)
        elif opt.optimizer == 'ada':
            self.optimizer_qq = optim.Adadelta(parameters, lr=opt.learning_rate)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)

        vocab_size = emb_vs.shape[0]
        assert self.opt.emb_dim == emb_vs.shape[1], print(emb_vs.shape)
        self.embedding_ly = nn.Embedding(vocab_size, self.opt.emb_dim, padding_idx=padding_idx)
        self.embedding_ly.weight.data = torch.FloatTensor(emb_vs)

        self.drop_emb = nn.Dropout(self.opt.dropout_emb)

        self.q_rep_ly = layers.StackedBRNNLSTM(
            input_size=opt.emb_dim,
            hidden_size=opt.q_hidden,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
        )

        self.rel_rep_ly = layers.StackedBRNNLSTM(
            input_size=opt.emb_dim,
            hidden_size=opt.q_hidden,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
        )
        self.q_r_match = layers.SeqAttnMatchNoMask(opt.q_hidden * 2)

        # self.o_liner_qr = nn.Linear(opt.merge_fts_dim * 2 * 2, 1, bias=True)
        self.o_liner_qr = nn.Linear(opt.merge_fts_dim * 2, 1, bias=True)

        if state_dict:
            new_state = set(self.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.load_state_dict(state_dict['network'])
        parameters = [p for p in self.parameters() if p.requires_grad]

        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, opt.learning_rate,
                                       momentum=opt.momentum,
                                       weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt.learning_rate,
                                          weight_decay=opt.weight_decay)
        elif opt.optimizer == 'ada':
            self.optimizer = optim.Adadelta(parameters, lr=opt.learning_rate)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)
        num_params = sum(p.data.numel() for p in parameters)
        self.logger.info("{} parameters".format(num_params))
        self.train_loss = AverageMeter()
        self.adv_train_loss = AverageMeter()

    def zero_loss(self):
        self.train_loss = AverageMeter()

    def forward(self, data, which='r', get_fts=False, given_fts=False):
        """Inputs:
        """
        if which == 'r':
            idxs2_rep = self.rel_rep_ly
            match_att_ly = self.q_r_match
            o_ly = self.o_liner_qr
        else:
            idxs2_rep = self.q_rep_ly
            # match_att_ly = self.q_q_match
            match_att_ly = self.q_r_match
            o_ly = self.o_liner_qq

        if given_fts:
            # merge_res1, merge_res2 = data
            # fts1 = F.max_pool1d(merge_res1.transpose(1, 2), kernel_size=merge_res1.size(1)).squeeze(-1)
            # fts2 = F.max_pool1d(merge_res2.transpose(1, 2), kernel_size=merge_res2.size(1)).squeeze(-1)
            # fts = torch.cat([fts1, fts2], dim=1)
            # return o_ly(fts)
            q_embs1, q_embs2 = data
        else:
            q_idxs1, q_idxs2 = data
            q_embs1 = self.drop_emb(self.embedding_ly(q_idxs1))
            q_embs2 = self.drop_emb(self.embedding_ly(q_idxs2))

        merge_sru = self.merge_sru_com
        q_hidden_states1 = self.q_rep_ly(q_embs1)

        q_hidden_states2 = idxs2_rep(q_embs2)

        match_state_for_1 = match_att_ly(q_hidden_states1, q_hidden_states2)
        # match_state_for_2 = self.q_q_match(q_hidden_states2, q_hidden_states1)

        # merge_in1 = torch.cat([q_hidden_states1,
        #                        match_state_for_1,
        #                        q_hidden_states1 * match_state_for_1], dim=2)
        merge_in1 = torch.cat([q_hidden_states1,
                               match_state_for_1], dim=2)
        merge_res1 = merge_sru(merge_in1)
        fts1 = F.max_pool1d(merge_res1.transpose(1, 2), kernel_size=merge_res1.size(1)).squeeze(-1)
        # 要套一下那个 capsule吗？

        # fts1 = F.max_pool1d(merge_res1.transpose(1, 2), kernel_size=merge_res1.size(1)).squeeze(-1)
        # fts = fts1
        # if which != 'r' or 1:
        # if which == 'r':
        return self.o_liner_qr(fts1)
        # match_state_for_2 = match_att_ly(q_hidden_states2, q_hidden_states1)
        # merge_in2 = torch.cat([q_hidden_states2,
        #                        match_state_for_2], dim=2)
        # merge_res2 = merge_sru(merge_in2)
        #
        # # merge_res = torch.cat([merge_res1, merge_res2], dim=1)
        # fts2 = F.max_pool1d(merge_res2.transpose(1, 2), kernel_size=merge_res2.size(1)).squeeze(-1)
        # fts = torch.cat([fts1, fts2], dim=1)
        #
        # probs = o_ly(fts)
        # if get_fts:
        #     # return probs, (merge_res1, merge_res2)
        #     return probs, (q_embs1, q_embs2)
        # return probs

    def to_cuda(self, np_array_list):
        v_list = [Variable(torch.from_numpy(e).long().cuda(async=True)) for e in np_array_list]
        return v_list

    def update(self, batch, which):
        self.train()
        batch = self.to_cuda(batch)
        q_idxs1, g_q_idxs2, neg_q_idxs2 = batch
        g_score = self((q_idxs1, g_q_idxs2), which=which[0])
        neg_score = self((q_idxs1, neg_q_idxs2), which=which[1])

        g_score = F.sigmoid(g_score)
        neg_score = F.sigmoid(neg_score)
        # loss = torch.sum(F.sigmoid(neg_score) - F.sigmoid(g_score))
        loss = F.margin_ranking_loss(g_score, neg_score,
                                     target=Variable(torch.ones(g_score.size())).cuda(),
                                     margin=0.5)
        # if which == 'qq':
        #     loss *= 1.5
        self.train_loss.update(loss.data[0], q_idxs1.size(0))

        # optimizer = self.optimizer_qq if which == 'qq' else self.optimizer
        optimizer = self.optimizer
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
        optimizer.step()

    def predict_score_of_batch(self, batch, which):
        self.eval()
        batch = self.to_cuda(batch)
        q_idxs1, q_idxs2, _ = batch
        score = self((q_idxs1, q_idxs2), which)
        score = F.sigmoid(score).squeeze()
        return score.data.cpu().numpy()

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                # 'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            self.logger.info('model saved to {}'.format(filename))
        except BaseException:
            self.logger.warning('[ WARN: Saving failed... continuing anyway. ]')

    def setup_loger(self):
        # setup logger
        log = logging.getLogger('TraditionalModel')
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


class TraditionalRDModel(nn.Module):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt, emb_vs, padding_idx=0, state_dict=None):
        super(TraditionalRDModel, self).__init__()
        self.opt = opt
        self.logger = self.setup_loger()

        emb_dim = self.opt.emb_dim
        vocab_size = emb_vs.shape[0]
        assert self.opt.emb_dim == emb_vs.shape[1], print(emb_vs.shape)
        self.char_embedding = nn.Embedding(vocab_size, self.opt.emb_dim, padding_idx=padding_idx)
        self.char_embedding.weight.data = torch.FloatTensor(emb_vs)
        emb_dim += 256
        self.drop_emb = nn.Dropout(self.opt.dropout_emb)

        self.rep_rnn_q = layers.StackedBRNN(
            input_size=emb_dim,
            hidden_size=opt.q_hidden,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net

        )
        self.rep_rnn_rel = layers.StackedBRNN(
            input_size=emb_dim,
            hidden_size=opt.rel_hidden,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net
        )
        # self.rep_rnn_rel_alias = layers.StackedBRNN(
        #     input_size=emb_dim,
        #     hidden_size=opt.rel_alias_hidden,
        #     num_layers=opt.sru_ly_num,
        #     dropout_rate=opt.dropout_rnn,
        #     dropout_output=opt.dropout_rnn_output,
        #     concat_layers=opt.concat_rnn_layers,
        #     res_net=opt.res_net
        # )

        self.q_rel_match = layers.SeqAttnMatchNoMask(opt.q_hidden*2)

        match_in_dim = opt.q_hidden * 2 * 2
        match_o_dim = opt.q_hidden * 2
        self.merge_liner = nn.Linear(match_in_dim, match_o_dim, bias=False)
        self.merge_sru = layers.StackedBRNN(
            # input_size=match_in_dim,
            input_size=match_o_dim,
            hidden_size=opt.merge_fts_dim,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            res_net=opt.res_net
        )

        self.o_liner = nn.Linear(opt.merge_fts_dim * 2, 1, bias=False)

        if state_dict:
            new_state = set(self.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.load_state_dict(state_dict['network'])
        parameters = [p for p in self.parameters() if p.requires_grad]

        if opt.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, opt.learning_rate,
                                       momentum=opt.momentum,
                                       weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters, opt.learning_rate,
                                          weight_decay=opt.weight_decay)
        elif opt.optimizer == 'ada':
            self.optimizer = optim.Adadelta(parameters, lr=opt.learning_rate)
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)
        num_params = sum(p.data.numel() for p in parameters)
        self.logger.info("TraditionalRDModel, {} parameters".format(num_params))
        self.train_loss = AverageMeter()
        self.adv_train_loss = AverageMeter()

    def zero_loss(self):
        self.train_loss = AverageMeter()

    def forward(self, data, only_get_fts=False):
        """Inputs:
        """
        q_idxs, rel_idxs, q_c_embs, rel_c_embs = data
        # Embed both document and question
        q_embs = self.char_embedding(q_idxs)
        q_embs = torch.cat([q_embs, q_c_embs], dim=2)

        q_embs = self.drop_emb(q_embs)
        q_hidden_states = self.rep_rnn_q(q_embs)

        rel_embs = self.char_embedding(rel_idxs)

        rel_embs = torch.cat([rel_embs, rel_c_embs], dim=2)
        rel_embs = self.drop_emb(rel_embs)
        rel_hidden_states = self.rep_rnn_rel(rel_embs)
        # print(rel_hidden_states.size())
        # print(q_hidden_states.size())
        rel_match_states = self.q_rel_match(q_hidden_states, rel_hidden_states)
        merge_in = torch.cat([rel_match_states, q_hidden_states], dim=2)
        merge_res = self.merge_sru(F.relu(self.merge_liner(merge_in)))

        fts = F.max_pool1d(merge_res.transpose(1, 2), kernel_size=merge_res.size(1)).squeeze(-1)
        if only_get_fts:
            return fts
        probs = self.o_liner(fts)
        return probs

    def update(self, batch):
        self.train()
        # q_idxs, g_rel_idxs, cdt_rel_idxs = batch
        q_idxs, g_rel_idxs, cdt_rel_idxs = [Variable(torch.from_numpy(e).long().cuda(async=True)) for e in
                                              batch[:3]]
        q_c_embs, g_r_c_embs, neg_r_c_embs = [Variable(torch.from_numpy(e), volatile=True).float().cuda() for e in batch[3:]]

        g_score = self((q_idxs, g_rel_idxs, q_c_embs, g_r_c_embs))
        neg_score = self((q_idxs, cdt_rel_idxs, q_c_embs, neg_r_c_embs))
        g_score = F.sigmoid(g_score)
        neg_score = F.sigmoid(neg_score)
        # loss = torch.sum(F.sigmoid(neg_score) - F.sigmoid(g_score))
        loss = F.margin_ranking_loss(g_score, neg_score,
                                     target=Variable(torch.ones(g_score.size())).cuda(),
                                     margin=0.5)
        self.train_loss.update(loss.data[0], len(q_idxs))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
        self.optimizer.step()

        acc = (g_score - neg_score).gt(0).float().sum() / q_idxs.size(0)
        return acc.data[0]

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                # 'updates': self.updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            self.logger.info('model saved to {}'.format(filename))
        except BaseException:
            self.logger.warning('[ WARN: Saving failed... continuing anyway. ]')

    def predict_score_of_batch(self, batch):
        self.eval()
        with torch.no_grad():
            q_idxs, rel_idxs, _ = [Variable(torch.from_numpy(e).long(), volatile=True).cuda() for e in batch[:3]]

            q_c_embs, r_c_embs =  [Variable(torch.from_numpy(e), volatile=True).float().cuda() for e in batch[3:]]
            score = self((q_idxs, rel_idxs, q_c_embs, r_c_embs))
            # score = F.sigmoid(score)
            score = score.view(-1)
            return score.data.cpu().numpy()

    def setup_loger(self):
        # setup logger
        log = logging.getLogger('TraditionalModel')
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


if __name__ == '__main__':
    pass


# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""

import logging
import sys
import random

import torch
import torch.nn as nn
import layers
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
from utils import AverageMeter


# class JointModel(nn.Module):
#     def __init__(self, opt, emb_vs, padding_idx=0, state_dict=None):
#         """
#         就是一个seq label 的model,
#         Embedding 层和RDModel share一个  不更新
#         """
#         super(JointModel, self).__init__()
#         self.opt = opt
#         self.logger = self.setup_loger()
#
#         self.q_q_match = layers.SeqAttnMatchNoMask(opt.q_hidden * 2)
#         match_in_dim = opt.q_hidden * 2 * (2 + 0)
#         self.merge_sru_com = layers.StackedBRNNLSTM(
#             input_size=match_in_dim,
#             hidden_size=opt.merge_fts_dim,
#             # num_layers=opt.sru_ly_num,
#             num_layers=1,
#             dropout_rate=opt.dropout_rnn,
#             dropout_output=opt.dropout_rnn_output,
#             concat_layers=opt.concat_rnn_layers
#         )
#
#         self.o_liner_qq = nn.Linear(opt.merge_fts_dim * 2 * 2, 1, bias=True)
#
#         parameters = [p for p in self.parameters() if p.requires_grad]
#
#         if opt.optimizer == 'sgd':
#             self.optimizer_qq = optim.SGD(parameters, opt.learning_rate,
#                                        momentum=opt.momentum,
#                                        weight_decay=opt.weight_decay)
#         elif opt.optimizer == 'adamax':
#             self.optimizer_qq = optim.Adamax(parameters, opt.learning_rate,
#                                           weight_decay=opt.weight_decay)
#         elif opt.optimizer == 'ada':
#             self.optimizer_qq = optim.Adadelta(parameters, lr=opt.learning_rate)
#         else:
#             raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)
#
#         vocab_size = emb_vs.shape[0]
#         assert self.opt.emb_dim == emb_vs.shape[1], print(emb_vs.shape)
#         self.embedding_ly = nn.Embedding(vocab_size, self.opt.emb_dim, padding_idx=padding_idx)
#         self.embedding_ly.weight.data = torch.FloatTensor(emb_vs)
#
#         self.drop_emb = nn.Dropout(self.opt.dropout_emb)
#
#         self.q_rep_ly = layers.StackedBRNNLSTM(
#             input_size=opt.emb_dim,
#             hidden_size=opt.q_hidden,
#             num_layers=opt.sru_ly_num,
#             dropout_rate=opt.dropout_rnn,
#             dropout_output=opt.dropout_rnn_output,
#             concat_layers=opt.concat_rnn_layers,
#         )
#
#         self.rel_rep_ly = layers.StackedBRNNLSTM(
#             input_size=opt.emb_dim,
#             hidden_size=opt.q_hidden,
#             num_layers=opt.sru_ly_num,
#             dropout_rate=opt.dropout_rnn,
#             dropout_output=opt.dropout_rnn_output,
#             concat_layers=opt.concat_rnn_layers,
#         )
#         self.q_r_match = layers.SeqAttnMatchNoMask(opt.q_hidden * 2)
#
#         # self.o_liner_qr = nn.Linear(opt.merge_fts_dim * 2 * 2, 1, bias=True)
#         self.o_liner_qr = nn.Linear(opt.merge_fts_dim * 2, 1, bias=True)
#
#         if state_dict:
#             new_state = set(self.state_dict().keys())
#             for k in list(state_dict['network'].keys()):
#                 if k not in new_state:
#                     del state_dict['network'][k]
#             self.load_state_dict(state_dict['network'])
#         parameters = [p for p in self.parameters() if p.requires_grad]
#
#         if opt.optimizer == 'sgd':
#             self.optimizer = optim.SGD(parameters, opt.learning_rate,
#                                        momentum=opt.momentum,
#                                        weight_decay=opt.weight_decay)
#         elif opt.optimizer == 'adamax':
#             self.optimizer = optim.Adamax(parameters, opt.learning_rate,
#                                           weight_decay=opt.weight_decay)
#         elif opt.optimizer == 'ada':
#             self.optimizer = optim.Adadelta(parameters, lr=opt.learning_rate)
#         else:
#             raise RuntimeError('Unsupported optimizer: %s' % opt.optimizer)
#         num_params = sum(p.data.numel() for p in parameters)
#         self.logger.info("{} parameters".format(num_params))
#         self.train_loss = AverageMeter()
#         self.adv_train_loss = AverageMeter()
#
#     def zero_loss(self):
#         self.train_loss = AverageMeter()
#
#     def forward(self, data, which='r', get_fts=False, given_fts=False):
#         """Inputs:
#         """
#         if which == 'r':
#             idxs2_rep = self.rel_rep_ly
#             match_att_ly = self.q_r_match
#             o_ly = self.o_liner_qr
#         else:
#             idxs2_rep = self.q_rep_ly
#             # match_att_ly = self.q_q_match
#             match_att_ly = self.q_r_match
#             o_ly = self.o_liner_qq
#
#         if given_fts:
#             # merge_res1, merge_res2 = data
#             # fts1 = F.max_pool1d(merge_res1.transpose(1, 2), kernel_size=merge_res1.size(1)).squeeze(-1)
#             # fts2 = F.max_pool1d(merge_res2.transpose(1, 2), kernel_size=merge_res2.size(1)).squeeze(-1)
#             # fts = torch.cat([fts1, fts2], dim=1)
#             # return o_ly(fts)
#             q_embs1, q_embs2 = data
#         else:
#             q_idxs1, q_idxs2 = data
#             q_embs1 = self.drop_emb(self.embedding_ly(q_idxs1))
#             q_embs2 = self.drop_emb(self.embedding_ly(q_idxs2))
#
#         merge_sru = self.merge_sru_com
#         q_hidden_states1 = self.q_rep_ly(q_embs1)
#
#         q_hidden_states2 = idxs2_rep(q_embs2)
#
#         match_state_for_1 = match_att_ly(q_hidden_states1, q_hidden_states2)
#         # match_state_for_2 = self.q_q_match(q_hidden_states2, q_hidden_states1)
#
#         # merge_in1 = torch.cat([q_hidden_states1,
#         #                        match_state_for_1,
#         #                        q_hidden_states1 * match_state_for_1], dim=2)
#         merge_in1 = torch.cat([q_hidden_states1,
#                                match_state_for_1], dim=2)
#         merge_res1 = merge_sru(merge_in1)
#         fts1 = F.max_pool1d(merge_res1.transpose(1, 2), kernel_size=merge_res1.size(1)).squeeze(-1)
#
#         # fts1 = F.max_pool1d(merge_res1.transpose(1, 2), kernel_size=merge_res1.size(1)).squeeze(-1)
#         # fts = fts1
#         # if which != 'r' or 1:
#         # if which == 'r':
#         return self.o_liner_qr(fts1)
#         # match_state_for_2 = match_att_ly(q_hidden_states2, q_hidden_states1)
#         # merge_in2 = torch.cat([q_hidden_states2,
#         #                        match_state_for_2], dim=2)
#         # merge_res2 = merge_sru(merge_in2)
#         #
#         # # merge_res = torch.cat([merge_res1, merge_res2], dim=1)
#         # fts2 = F.max_pool1d(merge_res2.transpose(1, 2), kernel_size=merge_res2.size(1)).squeeze(-1)
#         # fts = torch.cat([fts1, fts2], dim=1)
#         #
#         # probs = o_ly(fts)
#         # if get_fts:
#         #     # return probs, (merge_res1, merge_res2)
#         #     return probs, (q_embs1, q_embs2)
#         # return probs
#
#     def to_cuda(self, np_array_list):
#         v_list = [Variable(torch.from_numpy(e).long().cuda(async=True)) for e in np_array_list]
#         return v_list
#
#     def update(self, batch, which):
#         self.train()
#         batch = self.to_cuda(batch)
#         q_idxs1, g_q_idxs2, neg_q_idxs2 = batch
#         g_score = self((q_idxs1, g_q_idxs2), which=which[0])
#         neg_score = self((q_idxs1, neg_q_idxs2), which=which[1])
#
#         g_score = F.sigmoid(g_score)
#         neg_score = F.sigmoid(neg_score)
#         # loss = torch.sum(F.sigmoid(neg_score) - F.sigmoid(g_score))
#         loss = F.margin_ranking_loss(g_score, neg_score,
#                                      target=Variable(torch.ones(g_score.size())).cuda(),
#                                      margin=0.5)
#         # if which == 'qq':
#         #     loss *= 1.5
#         self.train_loss.update(loss.data[0], q_idxs1.size(0))
#
#         # optimizer = self.optimizer_qq if which == 'qq' else self.optimizer
#         optimizer = self.optimizer
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
#         optimizer.step()
#
#     def predict_score_of_batch(self, batch, which):
#         self.eval()
#         batch = self.to_cuda(batch)
#         q_idxs1, q_idxs2, _ = batch
#         score = self((q_idxs1, q_idxs2), which)
#         score = F.sigmoid(score).squeeze()
#         return score.data.cpu().numpy()
#
#     def save(self, filename, epoch):
#         params = {
#             'state_dict': {
#                 'network': self.state_dict(),
#                 'optimizer': self.optimizer.state_dict(),
#                 # 'updates': self.updates
#             },
#             'config': self.opt,
#             'epoch': epoch
#         }
#         try:
#             torch.save(params, filename)
#             self.logger.info('model saved to {}'.format(filename))
#         except BaseException:
#             self.logger.warning('[ WARN: Saving failed... continuing anyway. ]')
#
#     def setup_loger(self):
#         # setup logger
#         log = logging.getLogger('TraditionalModel')
#         log.setLevel(logging.DEBUG)
#         fh = logging.FileHandler(self.opt.log_file)
#         fh.setLevel(logging.DEBUG)
#         ch = logging.StreamHandler(sys.stdout)
#         ch.setLevel(logging.INFO)
#         formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
#         fh.setFormatter(formatter)
#         ch.setFormatter(formatter)
#         log.addHandler(fh)
#         log.addHandler(ch)
#         return log


class TraditionalRDModel(nn.Module):
    """Network for the Document Reader module of DrQA."""

    def __init__(self, opt, emb_vs, padding_idx=0, state_dict=None, wh_idxs=None):
        super(TraditionalRDModel, self).__init__()
        self.opt = opt
        self.whs_idxs = torch.Tensor([wh_idxs]).long().cuda()

        self.logger = self.setup_loger()

        emb_dim = self.opt.emb_dim
        vocab_size = emb_vs.shape[0]
        assert self.opt.emb_dim == emb_vs.shape[1], print(emb_vs.shape)
        self.char_embedding = nn.Embedding(vocab_size, self.opt.emb_dim, padding_idx=padding_idx)
        self.char_embedding.weight.data = torch.FloatTensor(emb_vs)

        self.drop_emb = nn.Dropout(self.opt.dropout_emb)

        self.rep_rnn_q = layers.StackedBRNN(
            input_size=emb_dim,
            hidden_size=opt.q_hidden,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            # res_net=opt.res_net

        )
        self.rep_rnn_rel = layers.StackedBRNN(
            input_size=emb_dim,
            hidden_size=opt.rel_hidden,
            num_layers=opt.sru_ly_num,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            # res_net=opt.res_net
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

        self.q_rel_match = layers.SeqAttnMatchNoMask(opt.q_hidden * 2)
        # self.rel_wh_match = layers.SeqAttnMatchNoMask(opt.emb_dim)

        match_in_dim = opt.q_hidden * 2 * 3
        match_o_dim = self.opt.match_o_dim
        # match_o_dim = opt.match_o_dim
        self.merge_liner = nn.Sequential(nn.Linear(match_in_dim, self.opt.match_o_dim, bias=False),
                                         nn.Dropout(opt.dropout_liner_output_rate),
                                         # nn.ReLU(),
                                         # nn.Linear(match_mid_dim, match_o_dim, bias=False),
                                         )
        self.merge_sru = layers.StackedBRNN(
            input_size=match_o_dim + opt.q_hidden * 2 * 2,
            # input_size=match_o_dim + opt.q_hidden * 0,
            hidden_size=opt.merge_fts_dim,
            num_layers=1,
            dropout_rate=opt.dropout_rnn,
            dropout_output=opt.dropout_rnn_output,
            concat_layers=opt.concat_rnn_layers,
            # res_net=opt.res_net
        )  # 给换成CNN 看看？
        # filter_nums=[50, 50 ,50 ]
        # self.merge_conv = layers.MyConv(emb_dim=match_o_dim + opt.q_hidden * 2,
        #                                 filter_nums=filter_nums,
        #                                 window_sizes=[1, 3, 5],
        #                                 cnn_out_drop=opt.dropout_rnn
        #                                 )
        self.o_liner = nn.Linear(opt.merge_fts_dim * 2, 1, bias=False)
        # self.o_liner = nn.Linear(sum(filter_nums), 1, bias=False)


        # self.final_merge = DynamicRouting(iter_num=3, out_cell_num=out_cell_num, in_dim=opt.merge_fts_dim * 2,
        #                                   out_dim=cell_dim)

        # self.o_liner = nn.Sequential(
        #     nn.Linear(ft_dim, opt.merge_fts_dim * 2, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(opt.dropout_liner),
        #     nn.Linear(opt.merge_fts_dim * 2, 1, bias=True)
        #     )

        self.type_emb_trans_liner = nn.Linear(opt.emb_dim, opt.q_hidden * 2, bias=False)
        # self.o_liner = nn.Linear(ft_dim, 1, bias=False)

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
        self.train_loss_not_null = AverageMeter()
        self.adv_train_loss = AverageMeter()

    def zero_loss(self):
        self.train_loss = AverageMeter()
        self.train_loss_not_null = AverageMeter()

    def forward(self, data, only_get_fts=False, adv_training=False, given_fts=False):
        """Inputs:
        """
        if given_fts:
            q_embs, rel_embs = data
        else:
            q_idxs, rel_idxs, wh_sts = data

            # Embed both document and question
            q_embs = self.char_embedding(q_idxs)
            q_embs = self.drop_emb(q_embs)

            rel_embs = self.char_embedding(rel_idxs)

        emb_dim = q_embs.size(2)
        if self.training:
            # print('pass')
            if random.randint(0, 100) > self.opt.drop_wh_sts:
                wh_sts = wh_sts * 0
            # if random.randint(0, 100) > 95:
            #     rel_embs = rel_embs[:, -1:]
            # elif random.randint(0, 100) > 95:
            #     rel_embs = rel_embs[:, :-1]
            # pass
        type_embs = self.char_embedding(Variable(self.whs_idxs))
        type_embs = type_embs.expand(q_embs.size(0), self.whs_idxs.size(1), emb_dim)

        rel_rep = rel_embs[:, -1:]

        # type_emb_vec = self.rel_wh_match(rel_rep, type_embs)

        # if self.training:
        #     # print('pass')
        #     if random.randint(0, 100) > 10:
        #         type_emb_vec = type_emb_vec * 0
        type_emb_vec_fix = layers.weighted_avg(type_embs, wh_sts).unsqueeze(1)
        # type_emb_vec = self.type_emb_trans_liner(type_emb_vec)
        type_emb_vec_fix = self.type_emb_trans_liner(type_emb_vec_fix)

        # # type_emb_vec = F.relu(type_emb_vec)
        # type_emb_vec = F.tanh(type_emb_vec)
        type_emb_vec_fix = F.tanh(type_emb_vec_fix)
        # rel_embs = torch.cat([type_emb_vec, rel_embs], dim=1)
        rel_embs = self.drop_emb(rel_embs)

        q_hidden_states = self.rep_rnn_q(q_embs)

        # rel_embs = torch.transpose(rel_embs, 0, 1)
        # rel_embs, type_embs = rel_embs[1:], rel_embs[:1]
        # rel_embs = torch.transpose(rel_embs, 0, 1)
        # type_embs = torch.transpose(type_embs, 0, 1).expand(batch_size, q_idxs.size(1), emb_dim)
        rel_hidden_states = self.rep_rnn_rel(rel_embs)
        if self.opt.with_wh:
            rel_hidden_states = torch.cat([type_emb_vec_fix, rel_hidden_states], dim=1)
            # rel_hidden_states = torch.cat([rel_hidden_states, type_emb_vec], dim=1)

        if self.opt.without_interactive:
            rel_match_states = F.max_pool1d(rel_hidden_states.transpose(1, 2),
                                            kernel_size=rel_hidden_states.size(1)).squeeze(-1).unsqueeze(1).expand(q_hidden_states.size())
        else:
            rel_match_states = self.q_rel_match(q_hidden_states, rel_hidden_states)

        # rel_match_states = F.max_pool1d(rel_hidden_states.transpose(1, 2),
        #                                 kernel_size=rel_hidden_states.size(1)).squeeze(-1).unsqueeze(1).expand(q_hidden_states.size())

        merge_in = torch.cat([rel_match_states, q_hidden_states], dim=2)
        # merge_in = torch.cat([merge_in, rel_match_states * q_hidden_states], dim=2)
        merge_in = torch.cat([merge_in, rel_match_states * q_hidden_states], dim=2)

        merge_in = self.merge_liner(merge_in)

        merge_in = F.relu(merge_in)
        merge_in = torch.cat([merge_in, rel_match_states, q_hidden_states], dim=2)
        merge_res = self.merge_sru(merge_in)
        fts = F.max_pool1d(merge_res.transpose(1, 2), kernel_size=merge_res.size(1)).squeeze(-1)
        # fts = self.merge_conv(merge_in)

        # fts_avg = F.avg_pool1d(merge_res.transpose(1, 2), kernel_size=merge_res.size(1)).squeeze(-1)
        # fts = self.final_merge(merge_res)
        # fts = torch.cat([fts, fts_avg], dim=1)
        if only_get_fts:
            return fts
        probs = self.o_liner(fts)
        if adv_training:
            return probs, (q_embs, rel_embs)
        return probs

    def update(self, batch):
        self.train()
        q_idxs, g_rel_idxs, cdt_rel_idxs, g_wh_sts, neg_wh_sts = batch
        q_idxs, g_rel_idxs, cdt_rel_idxs = [Variable(torch.from_numpy(e).long().cuda(async=True)) for e in
                                            (q_idxs, g_rel_idxs, cdt_rel_idxs)]
        g_wh_sts, neg_wh_sts = [Variable(torch.from_numpy(e).float().cuda(async=True)) for e in
                                (g_wh_sts, neg_wh_sts)]

        g_score = self((q_idxs, g_rel_idxs, g_wh_sts))
        neg_score = self((q_idxs, cdt_rel_idxs, neg_wh_sts))
        g_score = F.sigmoid(g_score)
        neg_score = F.sigmoid(neg_score)
        # loss = torch.sum(F.sigmoid(neg_score) - F.sigmoid(g_score))
        loss = F.margin_ranking_loss(g_score, neg_score,
                                     target=Variable(torch.ones(g_score.size())).cuda(),
                                     margin=0.5)
        loss_flag = F.margin_ranking_loss(g_score, neg_score,
                             target=Variable(torch.ones(g_score.size())).cuda(),
                             margin=0.5, reduce=False)> 0
        not_null = loss_flag.byte().sum().item() / q_idxs.size(0)

        self.train_loss_not_null.update(not_null, len(q_idxs))
        self.train_loss.update(loss.data[0], len(q_idxs))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
        self.optimizer.step()

        acc = (g_score - neg_score).gt(0).float().sum() / q_idxs.size(0)

        score_cost = 1 - (g_score - neg_score)
        score_cost = F.margin_ranking_loss(g_score, neg_score,
                                           target=Variable(torch.ones(g_score.size())).cuda(),
                                           margin=0.7, reduce=False)
        return acc.data[0], score_cost.view(-1).data.cpu().numpy()

    def adv_update(self, batch):
        self.train()
        q_idxs, g_rel_idxs, cdt_rel_idxs = batch
        q_idxs1, g_q_idxs2, neg_q_idxs2 = [Variable(torch.from_numpy(e).long().cuda(async=True)) for e in
                                           (q_idxs, g_rel_idxs, cdt_rel_idxs)]
        g_score, fts1 = self((q_idxs1, g_q_idxs2), adv_training=True)
        neg_score, fts2 = self((q_idxs1, neg_q_idxs2), adv_training=True)

        g_score = F.sigmoid(g_score)
        neg_score = F.sigmoid(neg_score)
        # loss = torch.sum(F.sigmoid(neg_score) - F.sigmoid(g_score))
        loss = F.margin_ranking_loss(g_score, neg_score,
                                     target=Variable(torch.ones(g_score.size())).cuda(),
                                     margin=0.5)

        grad_fts1 = grad(loss, fts1, retain_graph=True)
        grad_fts2 = grad(loss, fts2, retain_graph=True)
        # adv_rate = 0.01
        # adv_rate = 0.05
        adv_rate = self.opt.adv_rate

        adv_g_score = self(
            [adv_rate * Variable(F.normalize(grad_true_fts.data)).cuda() + true_fts for true_fts, grad_true_fts
             in zip(fts1, grad_fts1)], given_fts=True)
        adv_neg_score = self(
            [adv_rate * Variable(F.normalize(grad_true_fts.data)).cuda() + true_fts for true_fts, grad_true_fts
             in zip(fts2, grad_fts2)], given_fts=True)

        adv_g_score = F.sigmoid(adv_g_score)
        adv_neg_score = F.sigmoid(adv_neg_score)
        # loss = torch.sum(F.sigmoid(neg_score) - F.sigmoid(g_score))
        adv_loss = F.margin_ranking_loss(adv_g_score, adv_neg_score,
                                         target=Variable(torch.ones(adv_g_score.size())).cuda(),
                                         margin=0.5)

        self.train_loss.update(loss.data[0], q_idxs1.size(0))
        self.adv_train_loss.update(adv_loss.data[0], q_idxs1.size(0))

        total_loss = loss + adv_loss
        # optimizer = self.optimizer_qq if which == 'qq' else self.optimizer
        optimizer = self.optimizer
        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
        optimizer.step()
        acc = (g_score - neg_score).gt(0).float().sum() / q_idxs1.size(0)
        score_cost = F.margin_ranking_loss(g_score, neg_score,
                                           target=Variable(torch.ones(g_score.size())).cuda(),
                                           margin=0.6, reduce=False)
        return acc.data[0], score_cost.view(-1).data.cpu().numpy()

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
        with torch.no_grad():
            self.eval()
            q_idxs, rel_idxs = [Variable(torch.from_numpy(e).long(), volatile=True).cuda() for e in batch[:2]]
            wh_sts = Variable(torch.from_numpy(batch[2]).float(), volatile=True).cuda()
            score = self((q_idxs, rel_idxs, wh_sts))
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


class DynamicRouting(nn.Module):
    def __init__(self, iter_num, out_cell_num, in_dim, out_dim):
        super(DynamicRouting, self).__init__()
        self.out_cell_num = out_cell_num
        self.liner = nn.Sequential(nn.Linear(in_dim, out_dim * out_cell_num),
                                   # nn.Tanh(),
                                   )
        self.out_dim = out_dim
        self.iter_num = iter_num

    def forward(self, seq_reps):
        seq_len = seq_reps.size(1)
        batch_size = seq_reps.size(0)

        u_hat = self.liner(seq_reps)  # 不对吧?          '''shape(b_sz, caps_num, out_caps_num*out_caps_dim)'''
        u_hat = u_hat.view(batch_size, seq_len, self.out_cell_num, self.out_dim)

        B = torch.zeros([batch_size, seq_len, self.out_cell_num]).float().to('cuda')
        V = None
        for i in range(self.iter_num):
            C = F.softmax(B, dim=-1)  # 让in cell在各个输出cell上的和为1
            C = C.unsqueeze(-1)  # shape(b_sz, seq_len, out_caps_num, 1) # 考虑加mask
            weighted_uhat = C * u_hat  # shape(b_sz, seq_leen, out_caps_num, out_caps_dim)
            S = torch.sum(weighted_uhat, dim=1)  # shape(b_sz, out_caps_num, out_caps_dim)

            V = self.squash(S).unsqueeze(
                dim=1)  # shape(b_sz, out_caps_num, out_caps_dim) shape(b_sz, 1, out_caps_num, out_caps_dim)

            B = torch.sum(u_hat * V, dim=-1) + B

        V_ret = V.squeeze(dim=1).view(batch_size, -1)  # shape(b_sz, out_caps_num, out_caps_dim)
        return V_ret

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


if __name__ == '__main__':
    pass

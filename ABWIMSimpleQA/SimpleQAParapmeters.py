# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""


class RDParameters:
    def __init__(self, train_idx, dev):
        # self.train_batchsize = 256
        self.train_batchsize = 256
        self.valid_batchsize = 512

        # self.alias_num = 3
        self.alias_num = 1
        self.alias_q_max_len = 10
        self.match_o_dim = 900

        self.emb_dim = 300

        self.q_hidden = 150
        self.rel_hidden = self.q_hidden
        self.rel_alias_hidden = self.q_hidden
        # self.match_o_dim = int(self.q_hidden * )
        # self.match_o_dim = 900
        # self.match_mid_dim = self.q_hidden * 6

        self.merge_fts_dim = 250

        self.merge_kernel_sizes = [1, 2, 2]
        self.merge_filter_nums = [50, 50, 150]

        self.sru_ly_num = 1

        self.dropout_emb = 0.35
        self.dropout_rnn = 0.1
        self.dropout_rnn_output = True
        self.dropout_liner_output_rate = 0.25
        self.concat_rnn_layers = False
        self.res_net = False

        self.with_selection = True
        # self.with_selection = False
        self.with_wh = True
        self.with_only_dynamic_wh = False
        self.use_wh_hot = False

        self.without_interactive = False
        self.optimizer = 'adamax'
        # self.optimizer = 'ada'
        # self.learning_rate = 1
        self.learning_rate = 0.002
        self.adv_rate = 0.00025
        # self.learning_rate = 0.01
        # self.learning_rate = 0.1
        self.weight_decay = 0
        self.review_rate = 0.35
        # self.momentum = 0
        # self.reduce_lr = 0.1
        # train_idx = 'debug_keras'
        # train_idx = 'debug_keras_novel'
        self.train_idx = train_idx
        self.dev = dev
        self.log_file = 'logs/New_exp_idx_{}_on_{}.log'.format(train_idx, self.dev)
        self.model_dir = 'models/New_exp_idx_{}_on_{}/'.format(train_idx, self.dev)
        self.drop_wh_sts = 10
        self.resume_dsrc_flag = False

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


if __name__ == '__main__':
    pass


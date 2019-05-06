# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""


class JointWebqaParameters:
    def __init__(self, dev, train_idx=None):
        # self.train_batchsize = 128
        self.train_batchsize = 64
        self.valid_batchsize = 64

        self.emb_dim = 300
        # self.q_hidden = 250
        # self.q_hidden = 100
        self.q_hidden = 150
        self.rel_hidden = 150

        # self.merge_fts_dim = 200
        # self.merge_fts_dim = 200
        self.merge_fts_dim = 100
        # self.merge_fts_dim_qq = 150
        # self.merge_fts_dim_qq = 200

        # self.merge_fts_dim = 200
        # # self.merge_fts_dim_qq = 150
        # self.merge_fts_dim_qq = 200

        # self.sru_ly_num = 2
        self.sru_ly_num = 1

        # self.dropout_emb = 0.35
        # self.dropout_emb = 0.35
        self.dropout_emb = 0.35
        self.dropout_rnn = 0.1
        self.dropout_liner = 0.3
        self.dropout_rnn_output = True
        self.concat_rnn_layers = False
        self.res_net = False
        self.with_wh = True
        self.static_wh = True
        self.with_wh_dynamic = False
        self.use_wh_hot = False
        self.with_selection = True
        self.without_interactive = False

        # self.with_wh = True
        self.review_rate = 0.2   # default 0.2

        self.valid_q_max_num = 10  # 一个关系里面有超过10个关系，只取前十个

        self.optimizer = 'adamax'
        # self.optimizer = 'ada'
        # self.learning_rate = 1
        self.learning_rate = 0.001
        # self.learning_rate = 0.1
        self.weight_decay = 0
        self.momentum = 0
        # self.reduce_lr = 0.1
        # train_idx = 'RL_pretrain'
        if train_idx is None:
            train_idx = 'adv_tune_2'
        self.train_idx = train_idx
        self.dev = dev
        self.log_file = 'logs/New_exp_idx_{}_on_{}.log'.format(train_idx, self.dev)
        self.loss_file = 'logs/New_exp_idx_{}_on_{}.loss'.format(train_idx, self.dev)
        self.model_dir = 'models/New_exp_idx_{}_on_{}/'.format(train_idx, self.dev)

        self.resume_dsrc_flag = False
        self.trained_model = 'models/New_exp_idx_traditional_model_pytorch_on_0/model_epoch_100.h5.'

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value



if __name__ == '__main__':
    pass


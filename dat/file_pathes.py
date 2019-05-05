# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""
import os
import pickle

dp = os.path.abspath(os.path.dirname(__file__)) + '/'

common_used_static_log_f = dp + 'common_used_static.log'

# ########################################### SimpleQA #############################################
simple_qa_dp = u'/mydata/wp/SimpleQAMine/relations/'
simple_qa_rel_f = simple_qa_dp + u'relation.2M.list'

simple_qa_vocab_f = dp + 'SimpleQA_vocab.txt'

# ########################################### WebQA #############################################
web_q_f_dp = '/mydata/wp/SimpleQAMine/KBQA_RE_data/webqsp_relations/'
web_qa_rel_f = web_q_f_dp + 'relations.txt'

webqa_vocab_f = dp + 'WebQA_vocab.txt'

# ########################################### query log data #####################################
query_log_f = dp + 'all_query_res_sp.txt'
if __name__ == '__main__':
    pass

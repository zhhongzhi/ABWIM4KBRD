# -*- encoding: utf-8 -*-
"""
    @author: hongzhi
    @time: 
    @des: 
"""

lns = '''
WordVocab: Finished constructing vocabulary of 12102 total words.
72238 /mydata/wp/SimpleQAMine/relations/train.replace_ne.withpool
72238it [00:00, 2812111.42it/s]
72238
2790
1619
69448
72238 /mydata/wp/SimpleQAMine/relations/train.replace_ne.withpool
72238it [00:00, 2044357.77it/s]
72238 /mydata/wp/SimpleQAMine/relations/train.replace_ne.withpool
72238it [00:00, 2015392.97it/s]
72238 /mydata/wp/SimpleQAMine/relations/train.replace_ne.withpool
10309 /mydata/wp/SimpleQAMine/relations/valid.replace_ne.withpool
10309it [00:00, 36406.37it/s]
20609 /mydata/wp/SimpleQAMine/relations/test.replace_ne.withpool
20609it [00:00, 31352.03it/s]
72238it [00:01, 38429.28it/s]
06/27/2018 07:56:13 TraditionalRDModel, 6783900 parameters
06/27/2018 07:56:13 {
 "train_batchsize": 256,
 "valid_batchsize": 512,
 "alias_num": 1,
 "alias_q_max_len": 10,
 "emb_dim": 300,
 "q_hidden": 150,
 "rel_hidden": 150,
 "rel_alias_hidden": 150,
 "match_o_dim": 600,
 "merge_fts_dim": 250,
 "merge_kernel_sizes": [
  1,
  2,
  2
 ],
 "merge_filter_nums": [
  50,
  50,
  150
 ],
 "sru_ly_num": 1,
 "dropout_emb": 0.35,
 "dropout_rnn": 0.1,
 "dropout_rnn_output": true,
 "dropout_liner_output_rate": 0.0,
 "concat_rnn_layers": false,
 "res_net": false,
 "optimizer": "adamax",
 "learning_rate": 0.002,
 "adv_rate": 0.00025,
 "weight_decay": 0,
 "review_rate": 0.35,
 "train_idx": "find",
 "dev": 0,
 "log_file": "logs/New_exp_idx_find_on_0.log",
 "model_dir": "models/New_exp_idx_find_on_0/",
 "resume_dsrc_flag": false
}
499037 training sample num
/home/hongzhi/wp/KBQA_nlpcc_2018/ABWIMSimpleQA/JointModel.py:452: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  self.train_loss.update(loss.data[0], len(q_idxs))
/home/hongzhi/wp/KBQA_nlpcc_2018/ABWIMSimpleQA/JointModel.py:456: UserWarning: torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.
  torch.nn.utils.clip_grad_norm(self.parameters(), 20.0)
/home/hongzhi/wp/KBQA_nlpcc_2018/ABWIMSimpleQA/JointModel.py:465: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
  return acc.data[0], score_cost.view(-1).data.cpu().numpy()
06/27/2018 07:57:03 batch_idx=1000, loss=0.06056427210569382, adv_loss=0
06/27/2018 07:57:50 ================================= epoch: 0 ==================================
06/27/2018 07:57:50 loss=0.04818690940737724
06/27/2018 07:57:50 train_loss_not_null=69953.0, total_train_num=499037,
/home/hongzhi/wp/KBQA_nlpcc_2018/ABWIMSimpleQA/JointModel.py:538: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  q_idxs, rel_idxs = [Variable(torch.from_numpy(e).long(), volatile=True).cuda() for e in batch[:2]]
/home/hongzhi/wp/KBQA_nlpcc_2018/ABWIMSimpleQA/JointModel.py:539: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  wh_sts = Variable(torch.from_numpy(batch[2]).float(), volatile=True).cuda()
06/27/2018 07:57:52 new best valid f1 found
06/27/2018 07:57:52 valid, r_f1=0.9166747502182558, batch_num=1950, need
06/27/2018 07:57:56 test, r_f1=0.9140181474113251, batch_num=1950, need
499037 training sample num
06/27/2018 07:58:47 batch_idx=1000, loss=0.02803950384259224, adv_loss=0
06/27/2018 07:59:35 ================================= epoch: 1 ==================================
06/27/2018 07:59:35 loss=0.026387060061097145
06/27/2018 07:59:35 train_loss_not_null=42455.0, total_train_num=499037,
06/27/2018 07:59:37 new best valid f1 found
06/27/2018 07:59:37 valid, r_f1=0.9281210592686002, batch_num=3900, need
06/27/2018 07:59:41 test, r_f1=0.9253238876219128, batch_num=3900, need
499037 training sample num
selected indices=74026, review indices=148753
06/27/2018 08:00:33 batch_idx=1000, loss=0.022635528817772865, adv_loss=0
06/27/2018 08:01:20 ================================= epoch: 2 ==================================
06/27/2018 08:01:20 loss=0.021860972046852112
06/27/2018 08:01:20 train_loss_not_null=36254.0, total_train_num=499037,
06/27/2018 08:01:22 new best valid f1 found
06/27/2018 08:01:22 valid, r_f1=0.9319041614123581, batch_num=5850, need
06/27/2018 08:01:26 test, r_f1=0.9289630743849774, batch_num=5850, need
499037 training sample num
selected indices=65685, review indices=151673
06/27/2018 08:02:18 batch_idx=1000, loss=0.018877174705266953, adv_loss=0
06/27/2018 08:03:06 ================================= epoch: 3 ==================================
06/27/2018 08:03:06 loss=0.018828997388482094
06/27/2018 08:03:06 train_loss_not_null=31555.0, total_train_num=499037,
06/27/2018 08:03:08 new best valid f1 found
06/27/2018 08:03:08 valid, r_f1=0.9326801823649239, batch_num=7800, need
06/27/2018 08:03:12 test, r_f1=0.9297879567179388, batch_num=7800, need
499037 training sample num
selected indices=59304, review indices=153906
06/27/2018 08:04:03 batch_idx=1000, loss=0.01770760677754879, adv_loss=0
06/27/2018 08:04:51 ================================= epoch: 4 ==================================
06/27/2018 08:04:51 loss=0.0173493605107069
06/27/2018 08:04:51 train_loss_not_null=29483.0, total_train_num=499037,
06/27/2018 08:04:53 new best valid f1 found
06/27/2018 08:04:53 valid, r_f1=0.9333592006984188, batch_num=9750, need
06/27/2018 08:04:57 test, r_f1=0.9313891988936872, batch_num=9750, need
499037 training sample num
selected indices=55988, review indices=155067
06/27/2018 08:05:49 batch_idx=1000, loss=0.016438979655504227, adv_loss=0
06/27/2018 08:06:37 ================================= epoch: 5 ==================================
06/27/2018 08:06:37 loss=0.01619458571076393
06/27/2018 08:06:37 train_loss_not_null=27792.0, total_train_num=499037,
06/27/2018 08:06:39 new best valid f1 found
06/27/2018 08:06:39 valid, r_f1=0.9347172373654089, batch_num=11700, need
06/27/2018 08:06:43 test, r_f1=0.9333786209908292, batch_num=11700, need
499037 training sample num
selected indices=53478, review indices=155945
06/27/2018 08:07:25 ================================= epoch: 6 ==================================
06/27/2018 08:07:25 loss=0.01513311080634594
06/27/2018 08:07:25 train_loss_not_null=10894.0, total_train_num=209408,
06/27/2018 08:07:27 new best valid f1 found
06/27/2018 08:07:27 valid, r_f1=0.9353962556989038, batch_num=12518, need
06/27/2018 08:07:31 test, r_f1=0.933815323402397, batch_num=12518, need
499037 training sample num
selected indices=52628, review indices=156243
06/27/2018 08:08:13 ================================= epoch: 7 ==================================
06/27/2018 08:08:13 loss=0.014449985697865486
06/27/2018 08:08:13 train_loss_not_null=10686.0, total_train_num=208640,
06/27/2018 08:08:15 new best valid f1 found
06/27/2018 08:08:15 valid, r_f1=0.9357842661751867, batch_num=13333, need
06/27/2018 08:08:19 test, r_f1=0.9336212334417002, batch_num=13333, need
499037 training sample num
selected indices=52375, review indices=156331
06/27/2018 08:09:01 ================================= epoch: 8 ==================================
06/27/2018 08:09:01 loss=0.014046463184058666
06/27/2018 08:09:01 train_loss_not_null=10146.0, total_train_num=208640,
06/27/2018 08:09:03 new best valid f1 found
06/27/2018 08:09:03 valid, r_f1=0.936366281889611, batch_num=14148, need
06/27/2018 08:09:07 test, r_f1=0.9348828181862293, batch_num=14148, need
499037 training sample num
selected indices=51686, review indices=156572
06/27/2018 08:09:49 ================================= epoch: 9 ==================================
06/27/2018 08:09:49 loss=0.013834779150784016
06/27/2018 08:09:49 train_loss_not_null=10122.0, total_train_num=208128,
06/27/2018 08:09:51 new best valid f1 found
06/27/2018 08:09:51 valid, r_f1=0.9367542923658939, batch_num=14961, need
06/27/2018 08:09:55 test, r_f1=0.9345916832451842, batch_num=14961, need
499037 training sample num
selected indices=51499, review indices=156638
06/27/2018 08:10:37 ================================= epoch: 10 ==================================
06/27/2018 08:10:37 loss=0.014217445626854897
06/27/2018 08:10:37 train_loss_not_null=10366.0, total_train_num=208128,
06/27/2018 08:10:39 valid, r_f1=0.9357842661751867, batch_num=15774, need
06/27/2018 08:10:43 test, r_f1=0.935028385656752, batch_num=15774, need
499037 training sample num
selected indices=51878, review indices=156505
06/27/2018 08:11:25 ================================= epoch: 11 ==================================
06/27/2018 08:11:25 loss=0.013443906791508198
06/27/2018 08:11:25 train_loss_not_null=9965.0, total_train_num=208128,
06/27/2018 08:11:27 valid, r_f1=0.9366572897468232, batch_num=16587, need
06/27/2018 08:11:31 test, r_f1=0.9347372507157067, batch_num=16587, need
499037 training sample num
selected indices=51538, review indices=156624
06/27/2018 08:12:13 ================================= epoch: 12 ==================================
06/27/2018 08:12:13 loss=0.012973040342330933
06/27/2018 08:12:13 train_loss_not_null=9629.0, total_train_num=208128,
06/27/2018 08:12:15 valid, r_f1=0.9353962556989038, batch_num=17400, need
06/27/2018 08:12:19 test, r_f1=0.93454316075501, batch_num=17400, need
499037 training sample num
selected indices=51155, review indices=156758
06/27/2018 08:13:01 ================================= epoch: 13 ==================================
06/27/2018 08:13:01 loss=0.012890717014670372
06/27/2018 08:13:01 train_loss_not_null=9490.0, total_train_num=207872,
06/27/2018 08:13:03 valid, r_f1=0.9364632845086818, batch_num=18212, need
06/27/2018 08:13:07 test, r_f1=0.9343975932844873, batch_num=18212, need
499037 training sample num
selected indices=50940, review indices=156833
06/27/2018 08:13:49 ================================= epoch: 14 ==================================
06/27/2018 08:13:49 loss=0.012606428936123848
06/27/2018 08:13:49 train_loss_not_null=9231.0, total_train_num=207616,
06/27/2018 08:13:51 valid, r_f1=0.9367542923658939, batch_num=19023, need
06/27/2018 08:13:55 test, r_f1=0.9348828181862293, batch_num=19023, need
499037 training sample num
selected indices=50637, review indices=156940
06/27/2018 08:14:37 ================================= epoch: 15 ==================================
06/27/2018 08:14:37 loss=0.012444961816072464
06/27/2018 08:14:37 train_loss_not_null=9211.0, total_train_num=207360,
06/27/2018 08:14:38 new best valid f1 found
06/27/2018 08:14:38 valid, r_f1=0.938403336890096, batch_num=19833, need
06/27/2018 08:14:42 test, r_f1=0.9344946382648357, batch_num=19833, need
499037 training sample num
selected indices=50566, review indices=156964
06/27/2018 08:15:25 ================================= epoch: 16 ==================================
06/27/2018 08:15:25 loss=0.01233051810413599
06/27/2018 08:15:25 train_loss_not_null=9120.0, total_train_num=207360,
06/27/2018 08:15:26 valid, r_f1=0.9380153264138131, batch_num=20643, need
06/27/2018 08:15:30 test, r_f1=0.9346887282255325, batch_num=20643, need
499037 training sample num
selected indices=50720, review indices=156910
06/27/2018 08:16:12 ================================= epoch: 17 ==================================
06/27/2018 08:16:12 loss=0.011882735416293144
06/27/2018 08:16:12 train_loss_not_null=8864.0, total_train_num=207616,
06/27/2018 08:16:14 new best valid f1 found
06/27/2018 08:16:14 model saved to models/New_exp_idx_find_on_0/model_epoch_17.h5.
06/27/2018 08:16:14 valid, r_f1=0.9394703656998739, batch_num=21454, need
06/27/2018 08:16:18 test, r_f1=0.9359017904798874, batch_num=21454, need
499037 training sample num
selected indices=50359, review indices=157037
06/27/2018 08:17:00 ================================= epoch: 18 ==================================
06/27/2018 08:17:00 loss=0.011852039024233818
06/27/2018 08:17:00 train_loss_not_null=8778.0, total_train_num=207360,
06/27/2018 08:17:02 valid, r_f1=0.9376273159375303, batch_num=22264, need
06/27/2018 08:17:06 test, r_f1=0.9356591780290164, batch_num=22264, need
499037 training sample num
selected indices=50314, review indices=157053
06/27/2018 08:17:48 ================================= epoch: 19 ==================================
06/27/2018 08:17:48 loss=0.012007144279778004
06/27/2018 08:17:48 train_loss_not_null=8850.0, total_train_num=207360,
06/27/2018 08:17:50 valid, r_f1=0.9391793578426617, batch_num=23074, need
06/27/2018 08:17:54 test, r_f1=0.9357562230093649, batch_num=23074, need
499037 training sample num
selected indices=50151, review indices=157110
06/27/2018 08:18:36 ================================= epoch: 20 ==================================
06/27/2018 08:18:36 loss=0.011957372538745403
06/27/2018 08:18:36 train_loss_not_null=8656.0, total_train_num=207104,
06/27/2018 08:18:38 valid, r_f1=0.937724318556601, batch_num=23883, need
06/27/2018 08:18:42 test, r_f1=0.9355136105584939, batch_num=23883, need
499037 training sample num
selected indices=49845, review indices=157217
06/27/2018 08:19:24 ================================= epoch: 21 ==================================
06/27/2018 08:19:24 loss=0.011548268608748913
06/27/2018 08:19:24 train_loss_not_null=8473.0, total_train_num=206848,
06/27/2018 08:19:26 valid, r_f1=0.9386943447473082, batch_num=24691, need
06/27/2018 08:19:30 test, r_f1=0.9364355378718036, batch_num=24691, need
499037 training sample num
selected indices=49513, review indices=157333
06/27/2018 08:20:12 ================================= epoch: 22 ==================================
06/27/2018 08:20:12 loss=0.011672396212816238
06/27/2018 08:20:12 train_loss_not_null=8555.0, total_train_num=206592,
06/27/2018 08:20:14 valid, r_f1=0.9372393054612475, batch_num=25498, need
06/27/2018 08:20:18 test, r_f1=0.9359988354602358, batch_num=25498, need
499037 training sample num
selected indices=49772, review indices=157242
06/27/2018 08:21:00 ================================= epoch: 23 ==================================
06/27/2018 08:21:00 loss=0.011294832453131676
06/27/2018 08:21:00 train_loss_not_null=8436.0, total_train_num=206848,
06/27/2018 08:21:02 valid, r_f1=0.9379183237947425, batch_num=26306, need
06/27/2018 08:21:06 test, r_f1=0.9358047454995391, batch_num=26306, need
499037 training sample num
selected indices=49712, review indices=157263
06/27/2018 08:21:47 ================================= epoch: 24 ==================================
06/27/2018 08:21:47 loss=0.011199281550943851
06/27/2018 08:21:47 train_loss_not_null=8311.0, total_train_num=206848,
06/27/2018 08:21:49 valid, r_f1=0.937724318556601, batch_num=27114, need
06/27/2018 08:21:53 test, r_f1=0.9361929254209326, batch_num=27114, need
499037 training sample num
selected indices=49343, review indices=157392
06/27/2018 08:22:35 ================================= epoch: 25 ==================================
06/27/2018 08:22:35 loss=0.011377261951565742
06/27/2018 08:22:35 train_loss_not_null=8474.0, total_train_num=206592,
06/27/2018 08:22:37 valid, r_f1=0.9385003395091668, batch_num=27921, need
06/27/2018 08:22:41 test, r_f1=0.9355621330486681, batch_num=27921, need
499037 training sample num
selected indices=49528, review indices=157328
06/27/2018 08:23:23 ================================= epoch: 26 ==================================
06/27/2018 08:23:23 loss=0.011137614026665688
06/27/2018 08:23:23 train_loss_not_null=8304.0, total_train_num=206848,
06/27/2018 08:23:25 valid, r_f1=0.9378213211756717, batch_num=28729, need
06/27/2018 08:23:29 test, r_f1=0.9366296278325004, batch_num=28729, need
499037 training sample num
selected indices=49371, review indices=157383
06/27/2018 08:24:11 ================================= epoch: 27 ==================================
06/27/2018 08:24:11 loss=0.0109452148899436
06/27/2018 08:24:11 train_loss_not_null=8172.0, total_train_num=206592,
06/27/2018 08:24:13 valid, r_f1=0.9380153264138131, batch_num=29536, need
06/27/2018 08:24:17 test, r_f1=0.936775195303023, batch_num=29536, need
499037 training sample num
selected indices=49381, review indices=157379
06/27/2018 08:24:58 ================================= epoch: 28 ==================================
06/27/2018 08:24:58 loss=0.01096933800727129
06/27/2018 08:24:58 train_loss_not_null=8114.0, total_train_num=206592,
06/27/2018 08:25:00 valid, r_f1=0.9379183237947425, batch_num=30343, need
06/27/2018 08:25:04 test, r_f1=0.9368722402833713, batch_num=30343, need
499037 training sample num
selected indices=49362, review indices=157386
06/27/2018 08:25:46 ================================= epoch: 29 ==================================
06/27/2018 08:25:46 loss=0.010948505252599716
06/27/2018 08:25:46 train_loss_not_null=8213.0, total_train_num=206592,
06/27/2018 08:25:48 valid, r_f1=0.9390823552235911, batch_num=31150, need
06/27/2018 08:25:52 test, r_f1=0.9363384928914552, batch_num=31150, need
499037 training sample num
selected indices=49455, review indices=157353
06/27/2018 08:26:34 ================================= epoch: 30 ==================================
06/27/2018 08:26:34 loss=0.010860069654881954
06/27/2018 08:26:34 train_loss_not_null=8033.0, total_train_num=206592,
06/27/2018 08:26:36 valid, r_f1=0.9392763604617325, batch_num=31957, need
06/27/2018 08:26:40 test, r_f1=0.9367266728128487, batch_num=31957, need
499037 training sample num
selected indices=49271, review indices=157418
06/27/2018 08:27:21 ================================= epoch: 31 ==================================
06/27/2018 08:27:21 loss=0.011117472313344479
06/27/2018 08:27:21 train_loss_not_null=8237.0, total_train_num=206592,
06/27/2018 08:27:23 valid, r_f1=0.9390823552235911, batch_num=32764, need
06/27/2018 08:27:27 test, r_f1=0.9368722402833713, batch_num=32764, need
499037 training sample num
selected indices=49555, review indices=157318
06/27/2018 08:28:09 ================================= epoch: 32 ==================================
06/27/2018 08:28:09 loss=0.010944945737719536
06/27/2018 08:28:09 train_loss_not_null=8062.0, total_train_num=206848,
06/27/2018 08:28:11 valid, r_f1=0.938403336890096, batch_num=33572, need
06/27/2018 08:28:15 test, r_f1=0.936775195303023, batch_num=33572, need
499037 training sample num
selected indices=49416, review indices=157367
06/27/2018 08:28:57 ================================= epoch: 33 ==================================
06/27/2018 08:28:57 loss=0.010705871507525444
06/27/2018 08:28:57 train_loss_not_null=7973.0, total_train_num=206592,
06/27/2018 08:28:59 valid, r_f1=0.9388883499854496, batch_num=34379, need
06/27/2018 08:29:03 test, r_f1=0.9366781503226745, batch_num=34379, need
499037 training sample num
selected indices=49421, review indices=157365
06/27/2018 08:29:45 ================================= epoch: 34 ==================================
06/27/2018 08:29:45 loss=0.010913792066276073
06/27/2018 08:29:45 train_loss_not_null=8069.0, total_train_num=206592,
06/27/2018 08:29:47 new best valid f1 found
06/27/2018 08:29:47 model saved to models/New_exp_idx_find_on_0/model_epoch_34.h5.
06/27/2018 08:29:47 valid, r_f1=0.9400523814142981, batch_num=35186, need
06/27/2018 08:29:51 test, r_f1=0.9366781503226745, batch_num=35186, need
499037 training sample num
selected indices=49437, review indices=157360
06/27/2018 08:30:33 ================================= epoch: 35 ==================================
06/27/2018 08:30:33 loss=0.010613268241286278
06/27/2018 08:30:33 train_loss_not_null=7938.0, total_train_num=206592,
06/27/2018 08:30:35 valid, r_f1=0.9396643709380154, batch_num=35993, need
06/27/2018 08:30:39 test, r_f1=0.9369692852637197, batch_num=35993, need
499037 training sample num
selected indices=49240, review indices=157428
06/27/2018 08:31:21 ================================= epoch: 36 ==================================
06/27/2018 08:31:21 loss=0.010517425835132599
06/27/2018 08:31:21 train_loss_not_null=7948.0, total_train_num=206592,
06/27/2018 08:31:23 valid, r_f1=0.9389853526045203, batch_num=36800, need
06/27/2018 08:31:27 test, r_f1=0.936532582852152, batch_num=36800, need
499037 training sample num
selected indices=49315, review indices=157402
06/27/2018 08:32:09 ================================= epoch: 37 ==================================
06/27/2018 08:32:09 loss=0.010479874908924103
06/27/2018 08:32:09 train_loss_not_null=7767.0, total_train_num=206592,
06/27/2018 08:32:11 valid, r_f1=0.9392763604617325, batch_num=37607, need
06/27/2018 08:32:15 test, r_f1=0.9365811053423262, batch_num=37607, need
499037 training sample num
selected indices=48937, review indices=157535
06/27/2018 08:32:56 ================================= epoch: 38 ==================================
06/27/2018 08:32:56 loss=0.01027865894138813
06/27/2018 08:32:56 train_loss_not_null=7789.0, total_train_num=206336,
06/27/2018 08:32:58 valid, r_f1=0.9386943447473082, batch_num=38413, need
06/27/2018 08:33:02 test, r_f1=0.9370663302440682, batch_num=38413, need
499037 training sample num
selected indices=49052, review indices=157494
06/27/2018 08:33:44 ================================= epoch: 39 ==================================
06/27/2018 08:33:44 loss=0.010438235476613045
06/27/2018 08:33:44 train_loss_not_null=7819.0, total_train_num=206336,
06/27/2018 08:33:46 valid, r_f1=0.9388883499854496, batch_num=39219, need
06/27/2018 08:33:50 test, r_f1=0.9366781503226745, batch_num=39219, need
499037 training sample num
selected indices=48846, review indices=157566
06/27/2018 08:34:32 ================================= epoch: 40 ==================================
06/27/2018 08:34:32 loss=0.010018701665103436
06/27/2018 08:34:32 train_loss_not_null=7533.0, total_train_num=206336,
06/27/2018 08:34:34 valid, r_f1=0.938403336890096, batch_num=40025, need
06/27/2018 08:34:38 test, r_f1=0.936775195303023, batch_num=40025, need
499037 training sample num
selected indices=48572, review indices=157662
06/27/2018 08:35:20 ================================= epoch: 41 ==================================
06/27/2018 08:35:20 loss=0.010293672792613506
06/27/2018 08:35:20 train_loss_not_null=7755.0, total_train_num=206080,
06/27/2018 08:35:22 valid, r_f1=0.9380153264138131, batch_num=40830, need
06/27/2018 08:35:26 test, r_f1=0.9368722402833713, batch_num=40830, need
499037 training sample num
selected indices=48983, review indices=157518
06/27/2018 08:36:07 ================================= epoch: 42 ==================================
06/27/2018 08:36:07 loss=0.01026779692620039
06/27/2018 08:36:07 train_loss_not_null=7726.0, total_train_num=206336,
06/27/2018 08:36:09 valid, r_f1=0.938403336890096, batch_num=41636, need
06/27/2018 08:36:13 test, r_f1=0.9374059876752875, batch_num=41636, need
499037 training sample num
selected indices=48894, review indices=157550
06/27/2018 08:36:55 ================================= epoch: 43 ==================================
06/27/2018 08:36:55 loss=0.010276788845658302
06/27/2018 08:36:55 train_loss_not_null=7670.0, total_train_num=206336,
06/27/2018 08:36:57 valid, r_f1=0.9385973421282374, batch_num=42442, need
06/27/2018 08:37:01 test, r_f1=0.9372118977145907, batch_num=42442, need
499037 training sample num
selected indices=49140, review indices=157463
06/27/2018 08:37:43 ================================= epoch: 44 ==================================
06/27/2018 08:37:43 loss=0.010301955044269562
06/27/2018 08:37:43 train_loss_not_null=7730.0, total_train_num=206592,
06/27/2018 08:37:45 valid, r_f1=0.938403336890096, batch_num=43249, need
06/27/2018 08:37:49 test, r_f1=0.9374545101654617, batch_num=43249, need
499037 training sample num
selected indices=49219, review indices=157436
06/27/2018 08:38:31 ================================= epoch: 45 ==================================
06/27/2018 08:38:31 loss=0.010651614516973495
06/27/2018 08:38:31 train_loss_not_null=7924.0, total_train_num=206592,
06/27/2018 08:38:33 valid, r_f1=0.938403336890096, batch_num=44056, need
06/27/2018 08:38:37 test, r_f1=0.9373089426949391, batch_num=44056, need
499037 training sample num
selected indices=49336, review indices=157395
06/27/2018 08:39:19 ================================= epoch: 46 ==================================
06/27/2018 08:39:19 loss=0.010351195000112057
06/27/2018 08:39:19 train_loss_not_null=7647.0, total_train_num=206592,
06/27/2018 08:39:21 valid, r_f1=0.9385973421282374, batch_num=44863, need
06/27/2018 08:39:25 test, r_f1=0.9372604202047649, batch_num=44863, need
499037 training sample num
selected indices=49109, review indices=157474
06/27/2018 08:40:07 ================================= epoch: 47 ==================================
06/27/2018 08:40:07 loss=0.010075133293867111
06/27/2018 08:40:07 train_loss_not_null=7619.0, total_train_num=206336,
06/27/2018 08:40:09 valid, r_f1=0.9387913473663789, batch_num=45669, need
06/27/2018 08:40:13 test, r_f1=0.937017807753894, batch_num=45669, need
499037 training sample num
selected indices=48642, review indices=157638
06/27/2018 08:40:55 ================================= epoch: 48 ==================================
06/27/2018 08:40:55 loss=0.010187327861785889
06/27/2018 08:40:55 train_loss_not_null=7688.0, total_train_num=206080,
06/27/2018 08:40:57 valid, r_f1=0.9389853526045203, batch_num=46474, need
06/27/2018 08:41:01 test, r_f1=0.9373089426949391, batch_num=46474, need
499037 training sample num
selected indices=48945, review indices=157532
06/27/2018 08:41:43 ================================= epoch: 49 ==================================
06/27/2018 08:41:43 loss=0.010240423493087292
06/27/2018 08:41:43 train_loss_not_null=7621.0, total_train_num=206336,
06/27/2018 08:41:44 valid, r_f1=0.9387913473663789, batch_num=47280, need
06/27/2018 08:41:48 test, r_f1=0.9373574651851133, batch_num=47280, need
499037 training sample num
selected indices=48816, review indices=157577
06/27/2018 08:42:30 ================================= epoch: 50 ==================================
06/27/2018 08:42:30 loss=0.009966881945729256
06/27/2018 08:42:30 train_loss_not_null=7520.0, total_train_num=206336,
06/27/2018 08:42:32 valid, r_f1=0.9385973421282374, batch_num=48086, need
06/27/2018 08:42:36 test, r_f1=0.9370663302440682, batch_num=48086, need
499037 training sample num
selected indices=49045, review indices=157497
06/27/2018 08:43:18 ================================= epoch: 51 ==================================
06/27/2018 08:43:18 loss=0.010137282311916351
06/27/2018 08:43:18 train_loss_not_null=7640.0, total_train_num=206336,
06/27/2018 08:43:20 valid, r_f1=0.9385973421282374, batch_num=48892, need
06/27/2018 08:43:24 test, r_f1=0.9373574651851133, batch_num=48892, need
499037 training sample num
selected indices=48811, review indices=157579
06/27/2018 08:44:06 ================================= epoch: 52 ==================================
06/27/2018 08:44:06 loss=0.010138462297618389
06/27/2018 08:44:06 train_loss_not_null=7706.0, total_train_num=206336,
06/27/2018 08:44:08 valid, r_f1=0.9378213211756717, batch_num=49698, need
06/27/2018 08:44:12 test, r_f1=0.9372118977145907, batch_num=49698, need
499037 training sample num
selected indices=48976, review indices=157521
06/27/2018 08:44:54 ================================= epoch: 53 ==================================
06/27/2018 08:44:54 loss=0.01008697785437107
06/27/2018 08:44:54 train_loss_not_null=7668.0, total_train_num=206336,
06/27/2018 08:44:56 valid, r_f1=0.9386943447473082, batch_num=50504, need
06/27/2018 08:45:00 test, r_f1=0.9374059876752875, batch_num=50504, need
499037 training sample num
selected indices=48643, review indices=157637
06/27/2018 08:45:42 ================================= epoch: 54 ==================================
06/27/2018 08:45:42 loss=0.009762482717633247
06/27/2018 08:45:42 train_loss_not_null=7298.0, total_train_num=206080,
06/27/2018 08:45:44 valid, r_f1=0.9389853526045203, batch_num=51309, need
06/27/2018 08:45:48 test, r_f1=0.9376486001261585, batch_num=51309, need
499037 training sample num
selected indices=48193, review indices=157795
06/27/2018 08:46:30 ================================= epoch: 55 ==================================
06/27/2018 08:46:30 loss=0.01016121543943882
06/27/2018 08:46:30 train_loss_not_null=7671.0, total_train_num=205824,
06/27/2018 08:46:32 valid, r_f1=0.9398583761761568, batch_num=52113, need
06/27/2018 08:46:36 test, r_f1=0.9374059876752875, batch_num=52113, need
499037 training sample num
selected indices=49323, review indices=157399
06/27/2018 08:47:18 ================================= epoch: 56 ==================================
06/27/2018 08:47:18 loss=0.010289231315255165
06/27/2018 08:47:18 train_loss_not_null=7509.0, total_train_num=206592,
06/27/2018 08:47:20 valid, r_f1=0.9393733630808032, batch_num=52920, need
06/27/2018 08:47:24 test, r_f1=0.9372118977145907, batch_num=52920, need
499037 training sample num
selected indices=48906, review indices=157545
06/27/2018 08:48:05 ================================= epoch: 57 ==================================
06/27/2018 08:48:05 loss=0.010193247348070145
06/27/2018 08:48:05 train_loss_not_null=7711.0, total_train_num=206336,
06/27/2018 08:48:07 valid, r_f1=0.9396643709380154, batch_num=53726, need
06/27/2018 08:48:11 test, r_f1=0.9373574651851133, batch_num=53726, need
499037 training sample num
selected indices=49044, review indices=157497
06/27/2018 08:48:53 ================================= epoch: 58 ==================================
06/27/2018 08:48:53 loss=0.009899044409394264
06/27/2018 08:48:53 train_loss_not_null=7467.0, total_train_num=206336,
06/27/2018 08:48:55 valid, r_f1=0.9390823552235911, batch_num=54532, need
06/27/2018 08:48:59 test, r_f1=0.9375030326556358, batch_num=54532, need
499037 training sample num
selected indices=48691, review indices=157621
06/27/2018 08:49:41 ================================= epoch: 59 ==================================
06/27/2018 08:49:41 loss=0.009586061351001263
06/27/2018 08:49:41 train_loss_not_null=7390.0, total_train_num=206080,
06/27/2018 08:49:43 valid, r_f1=0.9396643709380154, batch_num=55337, need
06/27/2018 08:49:47 test, r_f1=0.9373574651851133, batch_num=55337, need
499037 training sample num
selected indices=48815, review indices=157577
06/27/2018 08:50:29 ================================= epoch: 60 ==================================
06/27/2018 08:50:29 loss=0.009741172194480896
06/27/2018 08:50:29 train_loss_not_null=7385.0, total_train_num=206336,
06/27/2018 08:50:31 valid, r_f1=0.9395673683189446, batch_num=56143, need
06/27/2018 08:50:35 test, r_f1=0.9373574651851133, batch_num=56143, need
499037 training sample num
selected indices=48731, review indices=157607
06/27/2018 08:51:17 ================================= epoch: 61 ==================================
06/27/2018 08:51:17 loss=0.010138052515685558
06/27/2018 08:51:17 train_loss_not_null=7685.0, total_train_num=206336,
06/27/2018 08:51:19 valid, r_f1=0.9393733630808032, batch_num=56949, need
06/27/2018 08:51:23 test, r_f1=0.9376486001261585, batch_num=56949, need
'''

import re


def parse_log(log_lns):
    log_lns = log_lns.split('\n')
    kept_lns = []
    valid_info = []
    test_info = []
    batch_smp_info = []
    for ln in log_lns:
            m1 = re.search('valid, r_f1=(0.\d*), batch_num=(\d*), need', ln)
            if m1:
                valid_info.append([m1.group(2), m1.group(1)])
                continue
            m2 = re.search('test, r_f1=(0.\d*), batch_num=(\d*), need', ln)
            if m2:
                test_info.append([m2.group(2), m2.group(1)])
                continue
            m3 = re.search('train_loss_not_null=(\d*).0, total_train_num=(\d*),', ln)
            if m3:
                batch_smp_info.append([m3.group(2), m3.group(1)])
                continue
    print()


if __name__ == '__main__':
    parse_log(lns)

import enum
import sys
import os
import cPickle
import json
import random
sys.path.append('../')

import numpy as np
import tensorflow as tf

import framework.model.module
from framework.model.module import Mode
import framework.model.trntst
import framework.model.data
import framework.util.caption.utility
import framework.impl.encoder.dnn

import decoder.rnn
import trntst_util
import service.fast_cider

VE = 'encoder'
VD = 'decoder'
CELL = decoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.dnn.Config()
    self.subcfgs[VD] = decoder.rnn.Config()
    self.reward_alpha = .5
    self.margin = .1
    self.num_neg = 16
    self.num_sample = 1

    self.strategy = 'beam'

    self.metric = 'cider'

  def _assert(self):
    assert self.subcfgs[VE].dim_output == self.subcfgs[VD].subcfgs[CELL].dim_hidden


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.val_loss = False
  cfg.monitor_iter = 100
  cfg.trn_batch_size = 64
  cfg.tst_batch_size = 64
  cfg.base_lr = 1e-5
  cfg.num_epoch = kwargs['num_epoch']
  cfg.reward_alpha = kwargs['reward_alpha']
  cfg.num_neg = kwargs['num_neg']
  cfg.num_sample = kwargs['num_sample']
  cfg.margin = kwargs['margin']

  enc = cfg.subcfgs[VE]
  enc.dim_fts = kwargs['dim_fts']
  enc.dim_output = kwargs['dim_hidden']
  enc.keepin_prob = kwargs['content_keepin_prob']

  dec = cfg.subcfgs[VD]
  dec.num_step = kwargs['num_step']
  dec.dim_input = kwargs['dim_input']
  dec.dim_hidden = kwargs['dim_hidden']

  cell = dec.subcfgs[CELL]
  cell.dim_hidden = kwargs['dim_hidden']
  cell.dim_input = kwargs['dim_input']
  cell.keepout_prob = kwargs['cell_keepout_prob']
  cell.keepin_prob = kwargs['cell_keepin_prob']

  return cfg


class Model(framework.model.module.AbstractPGModel):
  name_scope = 'vemd.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    ROLL_CAPTIONID = 'sample_captionids'
    ROLL_CAPTION_MASK = 'sample_caption_masks'
    PN_REWARD = 'pn_reward'
    PS_CIDER = 'ps_cider'
    PN_CIDER = 'pn_cider'
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    OUT_WID = 'out_wid'
    LOG_PROB = 'log_prob'
    BEAM_CUM_LOG_PROB = 'beam_cum_log_prob'
    BEAM_PRE = 'beam_pre'
    BEAM_END = 'beam_end'
    BASELINE_OUT_WID = 'greedy_out_wid'

  def _set_submods(self):
    return {
      VE: framework.impl.encoder.dnn.Encoder(self._config.subcfgs[VE]),
      VD: decoder.rnn.Decoder(self._config.subcfgs[VD]),
    }

  def _add_input_in_mode(self, mode):
    if mode == framework.model.module.Mode.TRN_VAL:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(self._config.subcfgs[VE].dim_fts)), name=self.InKey.FT.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)
        # trn only
        rollout_captionids = tf.placeholder(
          tf.int32, shape=(None, self._config.num_sample, self._config.subcfgs[VD].num_step), name=self.InKey.ROLL_CAPTIONID.value)
        rollout_caption_masks = tf.placeholder(
          tf.float32, shape=(None, self._config.num_sample, self.config.subcfgs[VD].num_step), name=self.InKey.ROLL_CAPTION_MASK.value)
        pn_reward = tf.placeholder(
          tf.float32, shape=(None, self._config.num_sample), name=self.InKey.PN_REWARD.value) # (num_pos*sample)
        ps_cider = tf.placeholder(
          tf.float32, shape=(None, self._config.num_sample), name=self.InKey.PS_CIDER.value)
        pn_cider = tf.placeholder(
          tf.float32, shape=(None, self._config.num_neg), name=self.InKey.PN_CIDER.value)

      return {
        self.InKey.FT: fts,
        self.InKey.ROLL_CAPTIONID: rollout_captionids,
        self.InKey.ROLL_CAPTION_MASK: rollout_caption_masks,
        self.InKey.PN_REWARD: pn_reward,
        self.InKey.PS_CIDER: ps_cider,
        self.InKey.PN_CIDER: pn_cider,
        self.InKey.IS_TRN: is_training,
      }
    else:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(self._config.subcfgs[VE].dim_fts)), name=self.InKey.FT.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)

      return {
        self.InKey.FT: fts,
        self.InKey.IS_TRN: is_training,
      }

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[VE]
    decoder = self.submods[VD]

    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.FT: in_ops[self.InKey.FT],
      encoder.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
    }, mode)
    ft_embed = out_ops[encoder.OutKey.EMBED] # (None, dim_output)

    def trn_val(ft_embed):
      # val
      with tf.variable_scope(self.name_scope):
        num_pos = tf.shape(ft_embed)[0]
        init_wid = tf.zeros((num_pos,), dtype=tf.int32)
      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, Mode.VAL, search_strategy='greedy')
      out_wid = out_ops[decoder.OutKey.OUT_WID]

      # sample
      with tf.variable_scope(self.name_scope):
        num_step = self._config.subcfgs[VD].num_step
        expanded_ft_embed = tf.tile(tf.expand_dims(ft_embed, 1), [1, self._config.num_sample, 1]) # (None, num_sample, dim_output)
        expanded_ft_embed = tf.reshape(expanded_ft_embed, (-1, self._config.subcfgs[VE].dim_output)) # (None*num_sample, dim_output)
        sample_captionid = in_ops[self.InKey.ROLL_CAPTIONID] # (None, num_sample, num_step)
        sample_captionid = tf.reshape(sample_captionid, (-1, num_step)) # (None*num_sample, num_step)

      vd_inputs = {
        decoder.InKey.FT: expanded_ft_embed,
        decoder.InKey.CAPTIONID: sample_captionid,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, Mode.TRN, is_trn=False)

      with tf.variable_scope(self.name_scope):
        rollout_log_prob = out_ops[decoder.OutKey.LOG_PROB] # (None*num_sample, num_step)
        rollout_log_prob = tf.reshape(rollout_log_prob, (-1, self._config.num_sample, num_step-1))

      return {
        self.OutKey.OUT_WID: out_wid,
        self.OutKey.LOG_PROB: rollout_log_prob,
      }

    def rollout(ft_embed):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(ft_embed)[0]
        init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode,
        search_strategy='greedy')
      baseline_out_wid = out_ops[decoder.OutKey.OUT_WID]

      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
        search_strategy='sample', num_sample=self._config.num_sample, topk=-1)
      roll_out_wid = out_ops[decoder.OutKey.OUT_WID]
      return {
        self.OutKey.BASELINE_OUT_WID: baseline_out_wid,
        self.OutKey.OUT_WID: roll_out_wid,
      }

    def tst(ft_embed):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(ft_embed)[0]
        init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode,
        search_strategy='beam')
      return {
        self.OutKey.OUT_WID: out_ops[decoder.OutKey.OUT_WID],
        self.OutKey.BEAM_CUM_LOG_PROB: out_ops[decoder.OutKey.BEAM_CUM_LOG_PROB],
        self.OutKey.BEAM_PRE: out_ops[decoder.OutKey.BEAM_PRE],
        self.OutKey.BEAM_END: out_ops[decoder.OutKey.BEAM_END],
      }

    delegate = {
      framework.model.module.Mode.TRN_VAL: trn_val,
      framework.model.module.Mode.ROLLOUT: rollout,
      framework.model.module.Mode.TST: tst,
    }
    return delegate[mode](ft_embed)

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      ps_ciders = self._inputs[self.InKey.PS_CIDER]
      pn_ciders = self._inputs[self.InKey.PN_CIDER] # (None, num_neg)
      # pn_cider = tf.nn.top_k(pn_ciders, k=self._config.num_hard_neg)
      # margin_rewards = tf.expand_dims(ps_ciders, 2) - tf.expand_dims(pn_ciders, 1) - self._config.margin
      # margin_rewards = tf.reduce_mean(margin_rewards, axis=2) # (None, num_sample)
      pn_cider = tf.reduce_logsumexp(100.*pn_ciders, 1, True) / 100.
      margin_rewards = ps_ciders - pn_cider - self._config.margin # (None, num_sample)

      reward = (1.0 - self._config.reward_alpha) * self._inputs[self.InKey.PN_REWARD] + self._config.reward_alpha * margin_rewards
      sample_log_prob = self._outputs[self.OutKey.LOG_PROB]
      sample_caption_mask = self._inputs[self.InKey.ROLL_CAPTION_MASK]
      sample_caption_mask = sample_caption_mask[:, :, 1:]
      surrogate_loss = -tf.expand_dims(reward, axis=2) * sample_log_prob * sample_caption_mask
      surrogate_loss = tf.reduce_sum(surrogate_loss) / tf.reduce_sum(sample_caption_mask)
      self.op2monitor['reward'] = tf.reduce_mean(reward)
      self.op2monitor['loss'] = surrogate_loss

    return surrogate_loss 

  def op_in_rollout(self, **kwargs):
    return {
      self.OutKey.BASELINE_OUT_WID: self._rollout_outputs[self.OutKey.BASELINE_OUT_WID],
      self.OutKey.OUT_WID: self._rollout_outputs[self.OutKey.OUT_WID],
    }

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
    }

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
      self.OutKey.BEAM_CUM_LOG_PROB: self._outputs[self.OutKey.BEAM_CUM_LOG_PROB],
      self.OutKey.BEAM_PRE: self._outputs[self.OutKey.BEAM_PRE],
      self.OutKey.BEAM_END: self._outputs[self.OutKey.BEAM_END],
    }


PathCfg = trntst_util.PathCfg


class TrnTst(framework.model.trntst.PGTrnTst):
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    framework.model.trntst.TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = framework.util.caption.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

    with open(path_cfg.groundtruth_file) as f:
      self.vid2captions = cPickle.load(f)
    self.cider_scorer = service.fast_cider.CiderScorer()
    self.cider_scorer.init_refs(self.vid2captions)

  def feed_data_and_rollout(self, data, sess):
    op_dict = self.model.op_in_rollout()

    feed_dict = self._construct_feed_dict_in_rollout(data)
    baseline_out_wids, roll_out_wids = sess.run(
      [op_dict[self.model.OutKey.BASELINE_OUT_WID], op_dict[self.model.OutKey.OUT_WID]], feed_dict=feed_dict)

    rollout_captionids, rollout_caption_masks = trntst_util.gen_captionid_masks_from_wids(roll_out_wids[:, :, :-1])

    vids = data['vids']
    if self.model_cfg.metric == 'cider':
      baseline_ciders = trntst_util.eval_cider_in_rollout(baseline_out_wids, vids, self.int2str, self.cider) # (None, 1)
      rollout_ciders = trntst_util.eval_cider_in_rollout(roll_out_wids, vids, self.int2str, self.cider) # (None, num_sample)
    elif self.model_cfg.metric == 'bcmr':
      baseline_ciders = trntst_util.eval_BCMR_in_rollout(baseline_out_wids, vids, self.int2str, self.cider, self.vid2captions) # (None, 1)
      rollout_ciders = trntst_util.eval_BCMR_in_rollout(roll_out_wids, vids, self.int2str, self.cider, self.vid2captions) # (None, num_sample)
    pos_rewards = rollout_ciders - baseline_ciders # (None, num_sample)

    data['pn_rewards'] = pos_rewards
    data['ps_ciders'] = rollout_ciders
    data['rollout_captionids'] = rollout_captionids
    data['rollout_caption_masks'] = rollout_caption_masks

    return data

  def _construct_feed_dict_in_rollout(self, data):
    fts = data['fts']
    batch_size = fts.shape[0]

    return {
      self.model.rollout_inputs[self.model.InKey.FT]: fts,
      self.model.rollout_inputs[self.model.InKey.IS_TRN]: False,
    }

  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.PN_REWARD]: data['pn_rewards'],
      self.model.inputs[self.model.InKey.PS_CIDER]: data['ps_ciders'],
      self.model.inputs[self.model.InKey.PN_CIDER]: data['neg_ciders'],
      self.model.inputs[self.model.InKey.ROLL_CAPTIONID]: data['rollout_captionids'],
      self.model.inputs[self.model.InKey.ROLL_CAPTION_MASK]: data['rollout_caption_masks'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    trntst_util.predict_and_eval_in_val(self, sess, tst_reader, metrics)

  def predict_in_tst(self, sess, tst_reader, predict_file):
    trntst_util.predict_in_tst(self, sess, tst_reader, predict_file, 'beam')


class TrnReader(framework.model.data.Reader):
  def __init__(self, num_neg, ft_files, vid_file, annotation_file, captionstr_file, word_file, metric):
    self.num_neg = num_neg
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.idxs = []
    self.vids = np.empty(0)
    self.num_caption = 0
    self.videoid2captions = {}
    self.metric = metric

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    self.vids = np.load(vid_file)

    self.num_caption = self.captionids.shape[0]
    self.idxs = range(self.num_caption)

    self.shuffled_idxs = range(self.num_caption)
    random.shuffle(self.shuffled_idxs)

    videoid2captions = cPickle.load(open(captionstr_file))
    for vid in self.vids:
      self.videoid2captions[vid] = videoid2captions[vid]

    self.cider_scorer = service.fast_cider.CiderScorer()
    self.cider_scorer.init_refs(self.videoid2captions)

    self.int2str = framework.util.caption.utility.CaptionInt2str(word_file)

  def num_record(self):
    return self.num_caption

  def yield_trn_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      pos_idxs = self.idxs[i:i+batch_size]
      pos_ft_idxs = set(self.ft_idxs[pos_idxs].tolist())

      pos_fts = self.fts[self.ft_idxs[pos_idxs]]
      pos_vids = self.vids[self.ft_idxs[pos_idxs]]

      idxs = range(self.num_caption)
      random.shuffle(idxs)

      neg_fts = []
      neg_captionids= []
      neg_caption_masks = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        if ft_idx not in pos_ft_idxs:
          neg_fts.append(self.fts[ft_idx])
          neg_captionids.append(self.captionids[idx])
          neg_caption_masks.append(self.caption_masks[idx])
          if len(neg_fts) == self.num_neg:
            break
      neg_fts = np.array(neg_fts, dtype=np.float32)
      neg_captionids = np.array(neg_captionids, dtype=np.int32)
      neg_caption_masks = np.array(neg_caption_masks, dtype=np.int32)

      neg_scores = []
      for pos_vid in pos_vids:
        vids = [pos_vid] * self.num_neg
        if self.metric == 'cider':
          scores = trntst_util.eval_cider_in_rollout(neg_captionids, vids, self.int2str, self.cider_scorer)
          neg_scores.append(scores)
        elif self.metric == 'bcmr':
          scores = trntst_util.eval_BCMR_in_rollout(neg_captionids, vids, self.int2str, self.cider_scorer, self.videoid2captions)
          neg_scores.append(scores)
      pn_scores = np.array(np_scores, dtype=np.float32)

      yield {
        'fts': pos_fts,
        'neg_ciders': pn_scores,
        'vids': pos_vids,
      }


Reader = trntst_util.Reader

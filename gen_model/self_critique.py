import enum
import sys
import os
import cPickle
import json
import random
sys.path.append('../')

import tensorflow as tf
import numpy as np

import framework.model.module
from framework.model.module import Mode
import framework.model.trntst
import framework.model.data
import framework.util.caption.utility
import framework.impl.encoder.dnn

import decoder.rnn
import trntst_util
import vevd
import service.fast_cider

VE = 'encoder'
VD = 'decoder'
CELL = decoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.dnn.Config()
    self.subcfgs[VD] = decoder.rnn.Config()

    self.alpha = 0.
    self.num_sample = 10
    self.search_strategy = 'beam'
    self.reward_metric = 'cider'

  def _assert(self):
    assert self.subcfgs[VE].dim_output == self.subcfgs[VD].subcfgs[CELL].dim_hidden


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.val_loss = False
  cfg.monitor_iter = 100
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 64
  cfg.base_lr = 1e-5
  cfg.num_epoch = kwargs['num_epoch']
  cfg.num_sample = kwargs['num_sample']
  cfg.reward_metric = kwargs['reward_metric']
  cfg.alpha = kwargs['alpha']

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
  name_scope = 'vevd.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    GT_CAPTIONID = 'groundtruth_captionids' # (None, max_caption_len)
    GT_CAPTIONMASK = 'groundtruth_caption_masks' # (None, max_caption_len)
    ROLL_CAPTIONID = 'rollout_captionids'
    ROLL_CAPTIONMASK = 'rollout_caption_masks'
    IS_TRN = 'is_training'
    REWARD = 'reward'

  class OutKey(enum.Enum):
    LOGIT = 'logit'
    ROLL_LOGPROB = 'roll_log_prob'
    OUT_WID = 'out_wid'
    BEAM_CUM_LOG_PROB = 'beam_cum_log_prob'
    BEAM_PRE = 'beam_pre'
    BEAM_END = 'beam_end'
    BASELINE_OUT_WID = 'greedy_out_wid'
    ROLL_OUT_WID = 'sample_out_wid'
    SAMPLE_LOG_PROB = 'sample_log_prob'

  def _set_submods(self):
    return {
      VE: framework.impl.encoder.dnn.Encoder(self._config.subcfgs[VE]),
      VD: decoder.rnn.Decoder(self._config.subcfgs[VD]),
    }

  def _add_input_in_mode(self, mode):
    dim_fts = self._config.subcfgs[VE].dim_fts
    num_step = self._config.subcfgs[VD].num_step

    if mode == Mode.TRN_VAL:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(dim_fts)), name=self.InKey.FT.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)
        # trn only
        gt_captionids = tf.placeholder(
          tf.int32, shape=(None, num_step), name=self.InKey.GT_CAPTIONID.value)
        gt_caption_masks = tf.placeholder(
          tf.float32, shape=(None, num_step), name=self.InKey.GT_CAPTIONMASK.value)
        roll_captionids = tf.placeholder(
          tf.int32, shape=(None, self._config.num_sample, num_step), name=self.InKey.ROLL_CAPTIONID.value)
        roll_caption_masks = tf.placeholder(
          tf.float32, shape=(None, self._config.num_sample, num_step), name=self.InKey.ROLL_CAPTIONMASK.value)
        reward = tf.placeholder(
          tf.float32, shape=(None, self._config.num_sample), name=self.InKey.REWARD.value)

        return {
          self.InKey.FT: fts,
          self.InKey.GT_CAPTIONID: gt_captionids,
          self.InKey.GT_CAPTIONMASK: gt_caption_masks,
          self.InKey.ROLL_CAPTIONID: roll_captionids,
          self.InKey.ROLL_CAPTIONMASK: roll_caption_masks,
          self.InKey.IS_TRN: is_training,
          self.InKey.REWARD: reward,
        }
    else:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(dim_fts)), name='fts')
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)

        return {
          self.InKey.FT: fts,
          self.InKey.IS_TRN: is_training,
        }

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode):
    num_sample = self._config.num_sample
    num_step = self._config.subcfgs[VD].num_step
    num_class = self._config.subcfgs[VD].num_words
    dim_output = self._config.subcfgs[VE].dim_output

    encoder = self.submods[VE]
    decoder = self.submods[VD]

    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.FT: in_ops[self.InKey.FT],
      encoder.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
    }, mode)
    ft_embed = out_ops[encoder.OutKey.EMBED] # (None, dim_output)

    def trn_val(ft_embed):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(ft_embed)[0]
        init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: in_ops[self.InKey.GT_CAPTIONID],
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
        is_trn=True, search_strategy='greedy')
      gt_logit = out_ops[decoder.OutKey.LOGIT]
      val_out_wid = out_ops[decoder.OutKey.OUT_WID]

      # reinforce ops
      with tf.variable_scope(self.name_scope):
        init_wid = tf.zeros((batch_size*num_sample,), dtype=tf.int32)
        ft_embed = tf.tile(tf.expand_dims(ft_embed, 1), [1, num_sample, 1])
        ft_embed = tf.reshape(ft_embed, [-1, dim_output]) # (None*num_sample, dim_output)
        roll_captionids = in_ops[self.InKey.ROLL_CAPTIONID]
        roll_captionids = tf.reshape(roll_captionids, (-1, num_step)) # (None*num_sample, num_step)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: roll_captionids,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
        is_trn=False)
      roll_log_prob = out_ops[decoder.OutKey.LOG_PROB]
      with tf.variable_scope(self.name_scope):
        roll_log_prob = tf.reshape(roll_log_prob, [-1, num_sample, num_step-1])

      return {
        self.OutKey.LOGIT: gt_logit,
        self.OutKey.ROLL_LOGPROB: roll_log_prob,
        self.OutKey.OUT_WID: val_out_wid,
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
        search_strategy='sample', topk=-1, num_sample=self._config.num_sample)
      roll_out_wid = out_ops[decoder.OutKey.OUT_WID]
      print roll_out_wid.get_shape

      return {
        self.OutKey.BASELINE_OUT_WID: baseline_out_wid,
        self.OutKey.ROLL_OUT_WID: roll_out_wid,
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
      Mode.TRN_VAL: trn_val,
      Mode.ROLLOUT: rollout,
      Mode.TST: tst,
    }
    return delegate[mode](ft_embed)

  def _add_loss(self):
    alpha = self._config.alpha

    with tf.variable_scope(self.name_scope):
      logits = self._outputs[self.OutKey.LOGIT] # (None*max_words_in_caption, num_words)
      xentropy_loss = framework.util.expanded_op.cross_entropy_loss_on_rnn_logits(
        self._inputs[self.InKey.GT_CAPTIONID], self._inputs[self.InKey.GT_CAPTIONMASK], logits)
      self.op2monitor['xentropy_loss'] = xentropy_loss

      reward = self._inputs[self.InKey.REWARD]
      roll_mask = self._inputs[self.InKey.ROLL_CAPTIONMASK]
      roll_mask = roll_mask[:, :, 1:]
      roll_log_probs = self._outputs[self.OutKey.ROLL_LOGPROB] # (None, num_sample, max_words_in_caption)
      surrogate_loss = -tf.expand_dims(reward, axis=2) * roll_log_probs * roll_mask
      surrogate_loss = tf.reduce_sum(surrogate_loss) / tf.reduce_sum(roll_mask)
      self.op2monitor['surrogate_loss'] = surrogate_loss
      self.op2monitor['reward'] = tf.reduce_mean(reward)

      loss_op = (1. - alpha) * xentropy_loss + alpha * surrogate_loss

    return loss_op

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
      self.OutKey.BEAM_CUM_LOG_PROB: self._outputs[self.OutKey.BEAM_CUM_LOG_PROB],
      self.OutKey.BEAM_PRE: self._outputs[self.OutKey.BEAM_PRE],
      self.OutKey.BEAM_END: self._outputs[self.OutKey.BEAM_END],
    }

  def op_in_val(self, **kwargs):
    op_dict = framework.model.module.AbstractModel.op_in_val(self)
    op_dict.update({
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
    })
    return op_dict

  def op_in_rollout(self, **kwargs):
    op_dict = {
      self.OutKey.BASELINE_OUT_WID: self._rollout_outputs[self.OutKey.BASELINE_OUT_WID],
      self.OutKey.ROLL_OUT_WID: self._rollout_outputs[self.OutKey.ROLL_OUT_WID],
    }
    return op_dict


PathCfg = trntst_util.PathCfg


class TrnTst(framework.model.trntst.PGTrnTst):
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    framework.model.trntst.TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = framework.util.caption.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

    gt_file = path_cfg.groundtruth_file
    with open(gt_file) as f:
      self.vid2captions = cPickle.load(f)
    self.cider_scorer = service.fast_cider.CiderScorer()
    self.cider_scorer.init_refs(self.vid2captions)

  def feed_data_and_rollout(self, data, sess):
    op_dict = self.model.op_in_rollout()

    feed_dict = self._construct_feed_dict_in_rollout(data)
    baseline_out_wids, roll_out_wids = sess.run(
      [
        op_dict[self.model.OutKey.BASELINE_OUT_WID], 
        op_dict[self.model.OutKey.ROLL_OUT_WID], 
      ],
      feed_dict=feed_dict)

    vids = data['vids']
    if self.model_cfg.reward_metric == 'cider':
      baselines = trntst_util.eval_cider_in_rollout(baseline_out_wids, vids, self.int2str, self.cider_scorer) # (None, 1)
      rewards = trntst_util.eval_cider_in_rollout(roll_out_wids, vids, self.int2str, self.cider_scorer) # (None, num_sample)
    elif self.model_cfg.reward_metric == 'bcmr':
      baselines = trntst_util.eval_BCMR_in_rollout(baseline_out_wids, vids, self.int2str, self.cider_scorer, self.vid2captions)
      rewards = trntst_util.eval_BCMR_in_rollout(roll_out_wids, vids, self.int2str, self.cider_scorer, self.vid2captions)
    print roll_out_wids.shape
    roll_caption_ids, roll_caption_masks = trntst_util.gen_captionid_masks_from_wids(roll_out_wids)

    data['rewards'] = rewards - baselines # (None, num_sample,)
    data['roll_caption_ids'] = roll_caption_ids # (None, num_sample, num_step)
    data['roll_caption_masks'] = roll_caption_masks # (None, num_sample, num_step)
    return data

  def _construct_feed_dict_in_rollout(self, data):
    fts = data['fts']
    batch_size = fts.shape[0]

    return {
      self.model.rollout_inputs[self.model.InKey.FT]: fts,
      self.model.rollout_inputs[self.model.InKey.IS_TRN]: False,
    }

  def _construct_feed_dict_in_trn(self, data):
    fts = data['fts']
    captionids = data['captionids']
    caption_masks = data['caption_masks']
    roll_caption_ids = data['roll_caption_ids']
    roll_caption_masks = data['roll_caption_masks']
    rewards = data['rewards']

    batch_size = fts.shape[0]
    return {
      self.model.inputs[self.model.InKey.FT]: fts,
      self.model.inputs[self.model.InKey.GT_CAPTIONID]: captionids,
      self.model.inputs[self.model.InKey.GT_CAPTIONMASK]: caption_masks,
      self.model.inputs[self.model.InKey.IS_TRN]: True,
      self.model.inputs[self.model.InKey.ROLL_CAPTIONID]: roll_caption_ids,
      self.model.inputs[self.model.InKey.ROLL_CAPTIONMASK]: roll_caption_masks,
      self.model.inputs[self.model.InKey.REWARD]: rewards,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    trntst_util.predict_and_eval_in_val(self, sess, tst_reader, metrics)

  def predict_in_tst(self, sess, tst_reader, predict_file):
    trntst_util.predict_in_tst(self, sess, tst_reader, predict_file, 'beam')


Reader = vevd.Reader

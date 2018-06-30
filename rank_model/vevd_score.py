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
import framework.model.trntst
import framework.model.data
import framework.util.caption.utility
import framework.impl.encoder.dnn
import decoder.rnn
import trntst_util


VE = 'encoder'
VD = 'decoder'
CELL = decoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.dnn.Config()
    self.subcfgs[VD] = decoder.rnn.Config()
    self.num_neg = 64
    self.max_margin = 0.5

  def _assert(self):
    assert self.subcfgs[VE].dim_output == self.subcfgs[VD].subcfgs[CELL].dim_hidden


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.val_loss = False
  cfg.monitor_iter = 100
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 64
  cfg.base_lr = 1e-4
  cfg.num_epoch = kwargs['num_epoch']
  cfg.num_neg = kwargs['num_neg']
  cfg.max_margin = kwargs['max_margin']

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


class Model(framework.model.module.AbstractModel):
  name_scope = 'vevd.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    CAPTIONID = 'captionids'
    CAPTION_MASK = 'caption_masks'
    DELTA = 'delta'
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    SIM = 'sim'
    LOG_PROB = 'log_prob'
    NLOG_PROB = 'neg_log_prob'

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
        captionids = tf.placeholder(
          tf.int32, shape=(None, self._config.subcfgs[VD].num_step), name=self.InKey.CAPTIONID.value)
        caption_masks = tf.placeholder(
          tf.float32, shape=(None, self.config.subcfgs[VD].num_step), name=self.InKey.CAPTION_MASK.value)
        deltas = tf.placeholder(
          tf.float32, shape=(None, self._config.num_neg), name=self.InKey.DELTA.value)

      return {
        self.InKey.FT: fts,
        self.InKey.IS_TRN: is_training,
        self.InKey.CAPTIONID: captionids,
        self.InKey.CAPTION_MASK: caption_masks,
        self.InKey.DELTA: deltas,
      }
    else:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(self._config.subcfgs[VE].dim_fts)), name=self.InKey.FT.value)
        captionids = tf.placeholder(
          tf.int32, shape=(None, self._config.subcfgs[VD].num_step), name=self.InKey.CAPTIONID.value)
        caption_masks = tf.placeholder(
          tf.float32, shape=(None, self._config.subcfgs[VD].num_step), name=self.InKey.CAPTION_MASK.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)

      return {
        self.InKey.FT: fts,
        self.InKey.CAPTIONID: captionids,
        self.InKey.CAPTION_MASK: caption_masks,
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
      batch_size = tf.shape(ft_embed)[0]

      # val
      caption_masks = in_ops[self.InKey.CAPTION_MASK]
      init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: in_ops[self.InKey.CAPTIONID][:-self._config.num_neg],
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, is_trn=False)
      log_prob = out_ops[decoder.OutKey.LOG_PROB]
      val_norm_log_prob = tf.reduce_sum(log_prob*caption_masks[:, 1:], axis=1) / \
        tf.reduce_sum(caption_masks[:, 1:], axis=1) # (None,)

      # pos
      caption_masks = in_ops[self.InKey.CAPTION_MASK][:-self._config.num_neg]
      init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: in_ops[self.InKey.CAPTIONID][:-self._config.num_neg],
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, is_trn=False)
      log_prob = out_ops[decoder.OutKey.LOG_PROB]
      norm_log_prob = tf.reduce_sum(log_prob*caption_masks[:, 1:], axis=1) / \
        tf.reduce_sum(caption_masks[:, 1:], axis=1) # (None,)

      # neg
      ft_embed = tf.tile(tf.expand_dims(ft_embed, 1), [1, self._config.num_neg, 1]) # (None, num_neg, dim_output)
      ft_embed = tf.reshape(ft_embed, (-1, self._config.subcfgs[VE].dim_output)) # (None*num_neg, dim_output)
      neg_captionid = in_ops[self.InKey.CAPTIONID][-self._config.num_neg:]
      neg_captionid = tf.tile(tf.expand_dims(neg_captionid, 0), [batch_size, 1, 1]) # (None, num_neg, num_step)
      neg_captionid = tf.reshape(neg_captionid, (-1, self._config.subcfgs[VD].num_step))
      neg_caption_masks = in_ops[self.InKey.CAPTION_MASK][-self._config.num_neg:]
      neg_caption_masks = tf.tile(tf.expand_dims(neg_caption_masks, 0), [batch_size, 1, 1]) # (None, num_neg, num_step)
      neg_caption_masks = tf.reshape(neg_caption_masks, (-1, self._config.subcfgs[VD].num_step))
      init_wid = tf.zeros((batch_size*self._config.num_neg,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: neg_captionid,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, is_trn=False)

      neg_log_prob = out_ops[decoder.OutKey.LOG_PROB]
      norm_neg_log_prob = tf.reduce_sum(neg_log_prob * neg_caption_masks[:, 1:], axis=1) / \
        tf.reduce_sum(neg_caption_masks[:, 1:], axis=1) # (None*num_neg,)
      norm_neg_log_prob = tf.reshape(norm_neg_log_prob, (-1, self._config.num_neg)) # (None, num_neg)

      return {
        self.OutKey.LOG_PROB: norm_log_prob,
        self.OutKey.NLOG_PROB: norm_neg_log_prob,
        self.OutKey.SIM: val_norm_log_prob,
      }

    def tst(ft_embed):
      batch_size = tf.shape(ft_embed)[0]
      init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.CAPTIONID: in_ops[self.InKey.CAPTIONID],
        decoder.InKey.INIT_WID: init_wid,
      }
      decoder.is_trn = False
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, task='retrieval')

      caption_masks = in_ops[self.InKey.CAPTION_MASK]
      log_prob = out_ops[decoder.OutKey.LOG_PROB]
      norm_log_prob = tf.reduce_sum(log_prob*caption_masks[:, 1:], axis=1) / \
        tf.reduce_sum(caption_masks[:, 1:], axis=1) # (None,)
      return {
        self.OutKey.SIM: norm_log_prob
      }

    delegate = {
      framework.model.module.Mode.TRN_VAL: trn_val,
      framework.model.module.Mode.TST: tst,
    }
    return delegate[mode](ft_embed)

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      log_prob = self._outputs[self.OutKey.LOG_PROB]
      log_prob = tf.expand_dims(log_prob, 1) # (None, 1)
      neg_log_prob = self._outputs[self.OutKey.NLOG_PROB] # (None, num_neg)

      deltas = self._inputs[self.InKey.DELTA]
      max_margin = self._config.max_margin * tf.ones_like(deltas, dtype=tf.float32)
      margin = tf.minimum(deltas, max_margin)
      loss_op = tf.reduce_logsumexp(100*(margin + neg_log_prob), axis=1) / 100.
      loss_op -= log_prob
      loss_op = tf.maximum(tf.zeros_like(loss_op), loss_op)
      loss_op = tf.reduce_mean(loss_op)
      self.op2monitor['loss'] = loss_op

    return loss_op

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.SIM: self._outputs[self.OutKey.SIM],
    }

  def op_in_tst(self):
    return {
      self.OutKey.SIM: self._outputs[self.OutKey.SIM],
    }


PathCfg = trntst_util.ScorePathCfg
TrnTst = trntst_util.ScoreTrnTst

TrnReader = trntst_util.ScoreTrnReader
ValReader = trntst_util.ValReader
TstReader = trntst_util.TstReader

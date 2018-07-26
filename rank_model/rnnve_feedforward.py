import os
import sys
import json
import cPickle
import random
import enum
sys.path.append('../')

import tensorflow as tf
import numpy as np

import framework.model.module
import framework.model.trntst
import framework.model.data
from framework.impl.encoder import dnn
import encoder.word
import encoder.birnn
import trntst_util

WE = 'word'
RNN = 'rnn'
CELL = encoder.birnn.CELL
RCELL = encoder.birnn.RCELL
FT_DNN = 'ft_dnn'
CAPTION_DNN = 'caption_dnn'


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[WE] = encoder.word.Config()
    self.subcfgs[RNN] = encoder.birnn.Config()
    self.subcfgs[FT_DNN] = dnn.Config()
    self.subcfgs[CAPTION_DNN] = dnn.Config()

    self.max_words_in_caption = 30
    self.pool = 'mean'

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = 0.1
    self.alpha = 0.5
    self.num_neg = 1
    self.l2norm = True

  def _assert(self):
    assert self.max_words_in_caption == self.subcfgs[RNN].num_step
    assert self.subcfgs[WE].dim_embed == self.subcfgs[RNN].subcfgs[CELL].dim_input
    assert self.subcfgs[FT_DNN].dim_output == self.dim_joint_embed
    assert sum(self.subcfgs[FT_DNN].dim_fts) == self.dim_ft
    assert self.subcfgs[CAPTION_DNN].dim_output == self.dim_joint_embed
    assert sum(self.subcfgs[CAPTION_DNN].dim_fts) == self.subcfgs[RNN].subcfgs[CELL].dim_hidden + self.subcfgs[RNN].subcfgs[RCELL].dim_hidden


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.val_loss = False
  cfg.monitor_iter = 100
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 64
  cfg.base_lr = 1e-4
  cfg.num_epoch = kwargs['num_epoch']

  cfg.margin = .1
  cfg.alpha = kwargs['alpha']
  cfg.num_neg = kwargs['num_neg']
  cfg.l2norm = kwargs['l2norm']
  cfg.dim_ft = kwargs['dim_ft']
  cfg.dim_joint_embed = kwargs['dim_joint_embed']

  cfg.max_words_in_caption = kwargs['max_words_in_caption']
  cfg.pool = kwargs['pool']

  we_cfg = cfg.subcfgs[WE]
  we_cfg.lr_mult = kwargs['lr_mult']
  we_cfg.dim_embed = kwargs['dim_word']

  rnn_cfg = cfg.subcfgs[RNN]
  rnn_cfg.num_step = kwargs['max_words_in_caption']
  rnn_cfg.cell_type = kwargs['cell']

  for cell in [CELL, RCELL]:
    cell_cfg = rnn_cfg.subcfgs[cell]
    cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
    cell_cfg.dim_input = kwargs['dim_word']
    cell_cfg.keepout_prob = 0.5
    cell_cfg.keepin_prob = 0.5

  ft_pca_cfg = cfg.subcfgs[FT_DNN]
  ft_pca_cfg.dim_fts = [cfg.dim_ft]
  ft_pca_cfg.dim_hiddens = kwargs['dim_ft_hiddens']
  ft_pca_cfg.dim_output = cfg.dim_joint_embed
  ft_pca_cfg.keepin_prob = kwargs['keepin_prob']

  caption_pca_cfg = cfg.subcfgs[CAPTION_DNN]
  caption_pca_cfg.dim_fts = [kwargs['cell_dim_hidden']*2]
  caption_pca_cfg.dim_hiddens = kwargs['dim_caption_hiddens']
  caption_pca_cfg.dim_output = cfg.dim_joint_embed
  caption_pca_cfg.keepin_prob = kwargs['keepin_prob']

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'rnnve_feedforward.Model'

  class InKey(enum.Enum):
    FT = 'ft'
    CAPTIONID = 'captionid'
    CAPTION_MASK = 'caption_mask'
    IS_TRN = 'is_trn'

  class OutKey(enum.Enum):
    SIM = 'sim'
    P_SIM = 'pos_sim'
    NF_SIM = 'neg_ft_sim'
    NC_SIM = 'neg_caption_sim'

  def _set_submods(self):
    ft_dnn = dnn.Encoder(self._config.subcfgs[FT_DNN])
    ft_dnn.name_scope = 'ft_dnn'
    caption_dnn = dnn.Encoder(self._config.subcfgs[CAPTION_DNN])
    caption_dnn.name_scope = 'caption_dnn'
    return {
      WE: encoder.word.Encoder(self._config.subcfgs[WE]),
      RNN: encoder.birnn.Encoder(self._config.subcfgs[RNN]),
      FT_DNN: ft_dnn,
      CAPTION_DNN: caption_dnn,
    }

  def _add_input_in_mode(self, mode):
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, self._config.dim_ft), name='fts')
      captionids = tf.placeholder(
        tf.int32, shape=(None, self._config.max_words_in_caption), name='captionids')
      caption_masks = tf.placeholder(
        tf.int32, shape=(None, self._config.max_words_in_caption), name='caption_masks')
      is_trn = tf.placeholder(tf.bool, shape=(), name='is_trn')

    return {
      self.InKey.FT: fts,
      self.InKey.CAPTIONID: captionids,
      self.InKey.CAPTION_MASK: caption_masks,
      self.InKey.IS_TRN: is_trn,
    }

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[WE]
    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.CAPTION: in_ops[self.InKey.CAPTIONID],
      }, mode)
    wvecs = out_ops[encoder.OutKey.EMBED] # (None, max_words_in_caption, dim_embed)

    with tf.variable_scope(self.name_scope):
      batch_size = tf.shape(wvecs)[0]
      dim_hidden = self._config.subcfgs[RNN].subcfgs[CELL].dim_hidden
      init_state = tf.zeros((batch_size, dim_hidden))

    rnn = self.submods[RNN]
    out_ops = rnn.get_out_ops_in_mode({
      rnn.InKey.FT: wvecs,
      rnn.InKey.MASK: in_ops[self.InKey.CAPTION_MASK],
      rnn.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      rnn.InKey.INIT_STATE: init_state,
    }, mode)
    caption = out_ops[rnn.OutKey.OUTPUT]

    with tf.variable_scope(self.name_scope):
      dim_input = self._config.subcfgs[RNN].subcfgs[CELL].dim_hidden + self._config.subcfgs[RNN].subcfgs[RCELL].dim_hidden
      caption = tf.reshape(caption, (-1, dim_input))
    caption_dnn = self.submods[CAPTION_DNN]
    out_ops = caption_dnn.get_out_ops_in_mode({
      caption_dnn.InKey.FT: caption,
      caption_dnn.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
    }, mode)
    caption_embed = out_ops[caption_dnn.OutKey.EMBED]
    with tf.variable_scope(self.name_scope):
      caption_embed = tf.reshape(caption_embed, (-1, self._config.max_words_in_caption, dim_input))
      mask = in_ops[self.InKey.CAPTION_MASK]
      mask = tf.expand_dims(tf.to_float(mask), 2)
      if self._config.pool == 'mean':
        caption_embed = tf.reduce_sum(caption_embed*mask, 1) / tf.reduce_sum(mask, 1)
      else:
        _mask = tf.cast(mask, tf.bool)
        _mask = tf.tile(_mask, [1, 1, tf.shape(caption_embed)[-1]])
        caption_embed = tf.where(_mask, caption_embed, -1e7*tf.ones_like(caption_embed, dtype=tf.float32))
        caption_embed = tf.reduce_max(caption_embed, 1)

    ft_dnn = self.submods[FT_DNN]
    out_ops = ft_dnn.get_out_ops_in_mode({
      ft_dnn.InKey.FT: in_ops[self.InKey.FT],
      ft_dnn.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
    }, mode)
    ft_embed = out_ops[ft_dnn.OutKey.EMBED]

    with tf.variable_scope(self.name_scope):
      if self._config.l2norm:
        caption_embed = tf.nn.l2_normalize(caption_embed, 1)
        ft_embed = tf.nn.l2_normalize(ft_embed, 1)    

    def trn(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        pos_ft_embed = ft_embed[:-self._config.num_neg]
        pos_caption_embed = caption_embed[:-self._config.num_neg]
        neg_ft_embed = ft_embed[-self._config.num_neg:]
        neg_caption_embed = caption_embed[-self._config.num_neg:]

        pos_sim = tf.reduce_sum(pos_ft_embed * pos_caption_embed, 1) # (trn_batch_size,)
        neg_caption_sim = tf.matmul(pos_ft_embed, neg_caption_embed, transpose_b=True) # (trn_batch_size, neg)
        neg_ft_sim = tf.matmul(pos_caption_embed, neg_ft_embed, transpose_b=True) # (trn_batch_size, neg)

        neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1) / 100. # (trn_batch_size,)
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1) / 100. # (trn_batch_size,)

      return pos_sim, neg_caption_sim, neg_ft_sim

    def tst(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        sim = tf.matmul(ft_embed, caption_embed, transpose_b=True)
      return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      pos_sim, neg_caption_sim, neg_ft_sim = trn(ft_embed, caption_embed)
      sim = tst(ft_embed, caption_embed)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_caption_sim,
      }
    else:
      sim = tst(ft_embed, caption_embed)
      return {
        self.OutKey.SIM: sim,
      }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      pos_sim = self._outputs[self.OutKey.P_SIM]
      neg_caption_sim = self._outputs[self.OutKey.NC_SIM]
      neg_ft_sim = self._outputs[self.OutKey.NF_SIM]

      contrast_caption_loss = neg_caption_sim + self._config.margin - pos_sim
      contrast_caption_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_caption_loss))
      self.op2monitor['contrast_caption_loss'] = tf.reduce_sum(contrast_caption_loss)

      contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
      contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))
      self.op2monitor['contrast_ft_loss'] = tf.reduce_sum(contrast_ft_loss)

      loss = self._config.alpha * contrast_caption_loss + (1.0 - self._config.alpha) * contrast_ft_loss
      loss = tf.reduce_sum(loss)
      self.op2monitor['loss'] = loss
    return loss

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.SIM: self._outputs[self.OutKey.SIM],
    }

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.SIM: self._outputs[self.OutKey.SIM],
    }


PathCfg = trntst_util.PathCfg
TrnTst = trntst_util.TrnTst

TrnReader = trntst_util.TrnReader
ValReader = trntst_util.ValReader
TstReader = trntst_util.TstReader

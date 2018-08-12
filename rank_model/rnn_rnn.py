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
import framework.impl.encoder.pca
import encoder.word
import encoder.birnn
import trntst_util

WE = 'word'
CAPTION_RNN = 'caption_rnn'
FT_RNN = 'ft_rnn'
CELL = encoder.birnn.CELL
RCELL = encoder.birnn.RCELL
FT_PCA = 'ft_pca'


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[WE] = encoder.word.Config()
    self.subcfgs[CAPTION_RNN] = encoder.birnn.Config()
    self.subcfgs[FT_RNN] = encoder.birnn.Config()
    self.subcfgs[FT_PCA] = framework.impl.encoder.pca.Config()

    self.max_words_in_caption = 30
    self.max_num_ft = 20
    self.pool_ft = 'mean'

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = 0.1
    self.alpha = 0.5
    self.num_neg = 1
    self.l2norm = True

  def _assert(self):
    assert self.max_words_in_caption == self.subcfgs[CAPTION_RNN].num_step
    assert self.subcfgs[WE].dim_embed == self.subcfgs[CAPTION_RNN].subcfgs[CELL].dim_input
    assert self.max_num_ft == self.subcfgs[FT_RNN].num_step
    assert self.subcfgs[FT_PCA].dim_output == self.subcfgs[FT_RNN].subcfgs[CELL].dim_input


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.val_loss = False
  cfg.monitor_iter = 100
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 64
  cfg.base_lr = 1e-4
  cfg.num_epoch = kwargs['num_epoch']

  cfg.margin = kwargs['margin']
  cfg.alpha = kwargs['alpha']
  cfg.num_neg = kwargs['num_neg']
  cfg.l2norm = kwargs['l2norm']
  cfg.dim_ft = kwargs['dim_ft']
  cfg.dim_joint_embed = kwargs['dim_joint_embed']

  cfg.max_words_in_caption = kwargs['max_words_in_caption']
  cfg.max_num_ft = kwargs['max_num_ft']
  cfg.pool_ft = kwargs['pool_ft']

  we_cfg = cfg.subcfgs[WE]
  we_cfg.lr_mult = kwargs['lr_mult']

  caption_rnn_cfg = cfg.subcfgs[CAPTION_RNN]
  caption_rnn_cfg.num_step = kwargs['max_words_in_caption']
  caption_rnn_cfg.cell_type = kwargs['cell']

  for cell in [CELL, RCELL]:
    cell_cfg = caption_rnn_cfg.subcfgs[cell]
    cell_cfg.dim_hidden = kwargs['caption_cell_dim_hidden']
    cell_cfg.dim_input = 300
    cell_cfg.keepout_prob = 0.5
    cell_cfg.keepin_prob = 0.5

  ft_pca_cfg = cfg.subcfgs[FT_PCA]
  ft_pca_cfg.dim_ft = kwargs['dim_ft']
  ft_pca_cfg.dim_output = kwargs['dim_pca_ft']

  ft_rnn_cfg = cfg.subcfgs[FT_RNN]
  ft_rnn_cfg.num_step = kwargs['max_num_ft']
  ft_rnn_cfg.cell_type = kwargs['cell']

  for cell in [CELL, RCELL]:
    cell_cfg = ft_rnn_cfg.subcfgs[cell]
    cell_cfg.dim_hidden = kwargs['ft_cell_dim_hidden']
    cell_cfg.dim_input = kwargs['dim_pca_ft']
    cell_cfg.keepout_prob = 0.5
    cell_cfg.keepin_prob = 0.5

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'rnnve.Model'

  class InKey(enum.Enum):
    FT = 'ft'
    FT_MASK = 'ft_mask'
    CAPTIONID = 'captionid'
    CAPTION_MASK = 'caption_mask'
    IS_TRN = 'is_trn'

  class OutKey(enum.Enum):
    SIM = 'sim'
    P_SIM = 'pos_sim'
    NF_SIM = 'neg_ft_sim'
    NC_SIM = 'neg_caption_sim'

  def _set_submods(self):
    caption_rnn = encoder.birnn.Encoder(self._config.subcfgs[CAPTION_RNN])
    caption_rnn.name_scope += '.caption'
    caption_rnn.submods[CELL].name_scope += '.caption'
    caption_rnn.submods[RCELL].name_scope += '.caption'
    ft_rnn = encoder.birnn.Encoder(self._config.subcfgs[FT_RNN])
    ft_rnn.name_scope += '.ft'
    ft_rnn.submods[CELL].name_scope += '.ft'
    ft_rnn.submods[RCELL].name_scope += '.ft'
    return {
      WE: encoder.word.Encoder(self._config.subcfgs[WE]),
      CAPTION_RNN: caption_rnn,
      FT_RNN: ft_rnn,
      FT_PCA: framework.impl.encoder.pca.Encoder1D(self._config.subcfgs[FT_PCA]),
    }

  def _add_input_in_mode(self, mode):
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, self._config.max_num_ft, self._config.dim_ft), name='fts')
      ft_masks = tf.placeholder(
        tf.float32, shape=(None, self._config.max_num_ft), name='ft_masks')
      captionids = tf.placeholder(
        tf.int32, shape=(None, self._config.max_words_in_caption), name='captionids')
      caption_masks = tf.placeholder(
        tf.int32, shape=(None, self._config.max_words_in_caption), name='caption_masks')
      is_trn = tf.placeholder(tf.bool, shape=(), name='is_trn')

    return {
      self.InKey.FT: fts,
      self.InKey.FT_MASK: ft_masks,
      self.InKey.CAPTIONID: captionids,
      self.InKey.CAPTION_MASK: caption_masks,
      self.InKey.IS_TRN: is_trn,
    }

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      dim_hidden = self._config.subcfgs[CAPTION_RNN].subcfgs[CELL].dim_hidden
      self.caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W',
        shape=(2*dim_hidden, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.caption_pca_B = tf.contrib.framework.model_variable('caption_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.caption_pca_W)
      self._weights.append(self.caption_pca_B)

      dim_hidden = self._config.subcfgs[FT_RNN].subcfgs[CELL].dim_hidden
      self.ft_pca_W = tf.contrib.framework.model_variable('ft_pca_W',
        shape=(2*dim_hidden, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.ft_pca_B = tf.contrib.framework.model_variable('ft_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.ft_pca_W)
      self._weights.append(self.ft_pca_B)

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    word_encoder = self.submods[WE]
    out_ops = word_encoder.get_out_ops_in_mode({
      word_encoder.InKey.CAPTION: in_ops[self.InKey.CAPTIONID],
      }, mode)
    wvecs = out_ops[word_encoder.OutKey.EMBED] # (None, max_words_in_caption, dim_embed)

    with tf.variable_scope(self.name_scope):
      batch_size = tf.shape(wvecs)[0]
      dim_hidden = self._config.subcfgs[CAPTION_RNN].subcfgs[CELL].dim_hidden
      init_state = tf.zeros((batch_size, dim_hidden))

    caption_rnn = self.submods[CAPTION_RNN]
    out_ops = caption_rnn.get_out_ops_in_mode({
      caption_rnn.InKey.FT: wvecs,
      caption_rnn.InKey.MASK: in_ops[self.InKey.CAPTION_MASK],
      caption_rnn.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      caption_rnn.InKey.INIT_STATE: init_state,
    }, mode)
    caption_embed = out_ops[caption_rnn.OutKey.OUTPUT]

    ft_pca = self.submods[FT_PCA]
    out_ops = ft_pca.get_out_ops_in_mode({
      ft_pca.InKey.FT: in_ops[self.InKey.FT],
      }, mode)
    fts = out_ops[ft_pca.OutKey.EMBED] # (None, max_num_ft, dim_pca_ft)

    with tf.variable_scope(self.name_scope):
      batch_size = tf.shape(fts)[0]
      dim_hidden = self._config.subcfgs[FT_RNN].subcfgs[CELL].dim_hidden
      init_state = tf.zeros((batch_size, dim_hidden))

    ft_rnn = self.submods[FT_RNN]
    out_ops = ft_rnn.get_out_ops_in_mode({
      ft_rnn.InKey.FT: fts,
      ft_rnn.InKey.MASK: in_ops[self.InKey.FT_MASK],
      ft_rnn.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      ft_rnn.InKey.INIT_STATE: init_state,
    }, mode)
    ft_embed = out_ops[ft_rnn.OutKey.OUTPUT]

    with tf.variable_scope(self.name_scope):
      mask = in_ops[self.InKey.CAPTION_MASK]
      mask = tf.expand_dims(tf.to_float(mask), 2)
      caption_embed = tf.nn.conv1d(caption_embed, tf.expand_dims(self.caption_pca_W, 0), 1, 'VALID')
      caption_embed = tf.nn.bias_add(caption_embed, self.caption_pca_B)
      caption_embed = tf.nn.tanh(caption_embed)
      caption_embed += 1.
      caption_embed = tf.reduce_max(caption_embed * mask, 1)
      caption_embed -= 1.
      if self._config.l2norm:
        caption_embed = tf.nn.l2_normalize(caption_embed, 1)

      mask = in_ops[self.InKey.FT_MASK]
      mask = tf.expand_dims(mask, 2)
      ft_embed = tf.nn.conv1d(ft_embed, tf.expand_dims(self.ft_pca_W, 0), 1, 'VALID')
      ft_embed = tf.nn.bias_add(ft_embed, self.ft_pca_B)
      ft_embed = tf.nn.tanh(ft_embed)
      if self._config.pool_ft == 'mean':
        ft_embed = tf.reduce_sum(ft_embed * mask, 1) / tf.reduce_sum(mask, 1)
      else:
        ft_embed += 1.
        ft_embed = tf.reduce_max(ft_embed * mask, 1)
        ft_embed -= 1.
      if self._config.l2norm:
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
TrnTst = trntst_util.AttTrnTst

TrnReader = trntst_util.TrnTemporalReader
ValReader = trntst_util.ValTemporalReader
TstReader = trntst_util.TstTemporalReader

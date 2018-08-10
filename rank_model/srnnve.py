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
import encoder.word
import framework.impl.encoder.rnn
import trntst_util
import rnnve

WE = 'word'
RNN = 'rnn'
CELL = framework.impl.encoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[WE] = encoder.word.Config()
    self.subcfgs[RNN] = framework.impl.encoder.rnn.RNNConfig()

    self.max_words_in_caption = 30

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = 0.1
    self.alpha = 0.5
    self.num_neg = 1
    self.l2norm = True
    self.pool = 'final'
    self.loss = 'lifted'

  def _assert(self):
    assert self.max_words_in_caption == self.subcfgs[RNN].num_step
    assert self.subcfgs[WE].dim_embed == self.subcfgs[RNN].subcfgs[CELL].dim_input


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

  rnn_cfg = cfg.subcfgs[RNN]
  rnn_cfg.num_step = kwargs['max_words_in_caption']
  rnn_cfg.cell_type = kwargs['cell']

  cell_cfg = rnn_cfg.subcfgs[CELL]
  cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
  cell_cfg.dim_input = 300
  cell_cfg.keepout_prob = 0.5
  cell_cfg.keepin_prob = 0.5

  return cfg


class Model(rnnve.Model):
  name_scope = 'srnnve.Model'

  def _set_submods(self):
    return {
      WE: encoder.word.Encoder(self._config.subcfgs[WE]),
      RNN: framework.impl.encoder.rnn.Encoder(self._config.subcfgs[RNN]),
    }

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      dim_hidden = self._config.subcfgs[RNN].subcfgs[CELL].dim_hidden
      if self._config.pool == 'mix':
        self.caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W',
          shape=(dim_hidden*2, self._config.dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
      else:
        self.caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W',
          shape=(dim_hidden, self._config.dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
      self.caption_pca_B = tf.contrib.framework.model_variable('caption_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.caption_pca_W)
      self._weights.append(self.caption_pca_B)

      self.ft_pca_W = tf.contrib.framework.model_variable('ft_pca_W',
        shape=(self._config.dim_ft, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.ft_pca_B = tf.contrib.framework.model_variable('ft_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.ft_pca_W)
      self._weights.append(self.ft_pca_B)

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
      rnn.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      rnn.InKey.INIT_STATE: init_state,
    }, mode)

    with tf.variable_scope(self.name_scope):
      outputs = out_ops[rnn.OutKey.OUTPUT]
      mask = in_ops[self.InKey.CAPTION_MASK]

      if self._config.pool == 'final':
        row_idxs = tf.range(batch_size)
        col_idxs = tf.to_int32(tf.reduce_sum(mask, 1))-1
        idx = tf.stack([row_idxs, col_idxs], axis=1) # (None, 2)

        output = tf.gather_nd(outputs, idx) # (None, dim_hidden)
        caption_embed = tf.nn.xw_plus_b(output, self.caption_pca_W, self.caption_pca_B)
        caption_embed = tf.nn.tanh(caption_embed)
      elif self._config.pool == 'max':
        mask = tf.expand_dims(tf.to_float(mask), 2)
        caption_embed = tf.nn.conv1d(outputs, tf.expand_dims(self.caption_pca_W, 0), 1, 'VALID')
        caption_embed = tf.nn.tanh(caption_embed)
        caption_embed += 1.
        caption_embed = tf.reduce_max(caption_embed * mask, 1)
        caption_embed -= 1.
      elif self._config.pool == 'mix':
        row_idxs = tf.range(batch_size)
        col_idxs = tf.to_int32(tf.reduce_sum(mask, 1))-1
        idx = tf.stack([row_idxs, col_idxs], axis=1) # (None, 2)
        output = tf.gather_nd(outputs, idx) # (None, dim_hidden)

        mask = tf.expand_dims(tf.to_float(mask), 2)
        base = tf.reduce_min(outputs, 1)
        outputs -= tf.expand_dims(base, 1)
        max_pool = tf.reduce_max(outputs * mask, 1)
        max_pool += base
        mix = tf.concat([output, max_pool], 1)
        mix = tf.nn.xw_plus_b(mix, self.caption_pca_W, self.caption_pca_B)
        caption_embed = tf.nn.tanh(mix)

      if self._config.l2norm:
        caption_embed = tf.nn.l2_normalize(caption_embed, 1)

      ft_embed = tf.nn.xw_plus_b(in_ops[self.InKey.FT], self.ft_pca_W, self.ft_pca_B)
      ft_embed = tf.nn.tanh(ft_embed)
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


PathCfg = trntst_util.PathCfg
TrnTst = trntst_util.TrnTst

TrnReader = trntst_util.TrnReader
ValReader = trntst_util.ValReader
TstReader = trntst_util.TstReader

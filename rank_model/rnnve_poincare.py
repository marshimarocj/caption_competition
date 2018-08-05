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
import encoder.birnn
import trntst_util
import framework.util.expanded_op

WE = 'word'
RNN = 'rnn'
CELL = encoder.birnn.CELL
RCELL = encoder.birnn.RCELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[WE] = encoder.word.Config()
    self.subcfgs[RNN] = encoder.birnn.Config()

    self.max_words_in_caption = 30
    self.pool = 'mean'

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = 1.
    self.alpha = 0.5
    self.num_neg = 1

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

  cfg.margin = 1.
  cfg.alpha = kwargs['alpha']
  cfg.num_neg = kwargs['num_neg']
  cfg.dim_ft = kwargs['dim_ft']
  cfg.dim_joint_embed = kwargs['dim_joint_embed']

  cfg.max_words_in_caption = kwargs['max_words_in_caption']
  cfg.pool = kwargs['pool']

  we_cfg = cfg.subcfgs[WE]
  we_cfg.lr_mult = kwargs['lr_mult']

  rnn_cfg = cfg.subcfgs[RNN]
  rnn_cfg.num_step = kwargs['max_words_in_caption']
  rnn_cfg.cell_type = kwargs['cell']

  for cell in [CELL, RCELL]:
    cell_cfg = rnn_cfg.subcfgs[cell]
    cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
    cell_cfg.dim_input = 300
    cell_cfg.keepout_prob = 0.5
    cell_cfg.keepin_prob = 0.5

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'rnnve_poincare.Model'

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
    return {
      WE: encoder.word.Encoder(self._config.subcfgs[WE]),
      RNN: encoder.birnn.Encoder(self._config.subcfgs[RNN]),
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
    with tf.variable_scope(self.name_scope):
      dim_hidden = self._config.subcfgs[RNN].subcfgs[CELL].dim_hidden
      self.caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W',
        shape=(2*dim_hidden, self._config.dim_joint_embed), dtype=tf.float32,
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
      rnn.InKey.MASK: in_ops[self.InKey.CAPTION_MASK],
      rnn.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      rnn.InKey.INIT_STATE: init_state,
    }, mode)

    with tf.variable_scope(self.name_scope):
      caption_embed = out_ops[rnn.OutKey.OUTPUT]
      mask = in_ops[self.InKey.CAPTION_MASK]
      mask = tf.expand_dims(tf.to_float(mask), 2)
      caption_embed = tf.nn.conv1d(caption_embed, tf.expand_dims(self.caption_pca_W, 0), 1, 'VALID')
      caption_embed = tf.nn.tanh(caption_embed)
      if self._config.pool == 'mean':
        caption_embed = tf.reduce_sum(caption_embed*mask, 1) / tf.reduce_sum(mask, 1)
      else:
        caption_embed += 1.
        caption_embed = tf.reduce_max(caption_embed * mask, 1)
        caption_embed -= 1.
      # unit ball
      caption_embed /= self._config.dim_joint_embed**0.5
      caption_embed = tf.clip_by_norm(caption_embed, 1.0, 1)
      caption_embed = framework.util.expanded_op.poincareball_gradient(caption_embed)

      ft_embed = tf.nn.xw_plus_b(in_ops[self.InKey.FT], self.ft_pca_W, self.ft_pca_B)
      ft_embed = tf.nn.tanh(ft_embed)
      # unit ball
      ft_embed /= self._config.dim_joint_embed**0.5
      ft_embed = tf.clip_by_norm(ft_embed, 1.0, 1)
      ft_embed = framework.util.expanded_op.poincareball_gradient(ft_embed)

    def trn(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        pos_ft_embed = ft_embed[:-self._config.num_neg]
        pos_caption_embed = caption_embed[:-self._config.num_neg]
        neg_ft_embed = ft_embed[-self._config.num_neg:]
        neg_caption_embed = caption_embed[-self._config.num_neg:]

        pos_dist = tf.square(tf.norm(pos_ft_embed - pos_caption_embed, axis=-1))
        pos_dist /= (1. - tf.square(tf.norm(pos_ft_embed, axis=-1))) * (1. - tf.square(tf.norm(pos_caption_embed, axis=-1)))
        pos_dist = 1 + 2 * pos_dist + 1e-6
        pos_dist = tf.acosh(pos_dist)
        pos_sim = -pos_dist

        neg_caption_dist = tf.square(tf.norm(tf.expand_dims(pos_ft_embed, 1) - tf.expand_dims(neg_caption_embed, 0), axis=-1))
        neg_caption_dist /= 1. - tf.square(tf.norm(tf.expand_dims(pos_ft_embed, 1), axis=-1))
        neg_caption_dist /= 1. - tf.square(tf.norm(tf.expand_dims(neg_caption_embed, 0), axis=-1))
        neg_caption_dist = 1 + 2 * neg_caption_dist + 1e-6
        neg_caption_dist = tf.acosh(neg_caption_dist)
        neg_caption_sim = -neg_caption_dist
        neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1) / 100.

        neg_ft_dist = tf.square(tf.norm(tf.expand_dims(pos_caption_embed, 1) - tf.expand_dims(neg_ft_embed, 0), axis=-1))
        neg_ft_dist /= 1. - tf.square(tf.norm(tf.expand_dims(pos_caption_embed, 1), axis=-1))
        neg_ft_dist /= 1. - tf.square(tf.norm(tf.expand_dims(neg_ft_embed, 0), axis=-1))
        neg_ft_dist = 1 + 2 * neg_ft_dist + 1e-6
        neg_ft_dist = tf.acosh(neg_ft_dist)
        neg_ft_sim = -neg_ft_dist
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1) / 100.

      return pos_sim, neg_caption_sim, neg_ft_sim

    def tst(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        dist = tf.square(tf.norm(tf.expand_dims(ft_embed, 1) - tf.expand_dims(caption_embed, 0), axis=-1))
        dist /= 1. - tf.square(tf.norm(tf.expand_dims(ft_embed, 1), axis=-1))
        dist /= 1. - tf.square(tf.norm(tf.expand_dims(caption_embed, 0), axis=-1))
        dist = 1 + 2 * dist + 1e-6
        dist = tf.acosh(dist)
        sim = -dist
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
      self.op2monitor['contrast_caption_loss'] = tf.reduce_mean(contrast_caption_loss)

      contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
      contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))
      self.op2monitor['contrast_ft_loss'] = tf.reduce_mean(contrast_ft_loss)

      loss = self._config.alpha * contrast_caption_loss + (1.0 - self._config.alpha) * contrast_ft_loss
      loss = tf.reduce_mean(loss)
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

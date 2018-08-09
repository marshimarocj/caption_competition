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
RNN = 'rnn'
CELL = encoder.birnn.CELL
RCELL = encoder.birnn.RCELL
TXT_EMBED = 'caption_embed'
FT_EMBED = 'ft_embed'


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[WE] = encoder.word.Config()
    self.subcfgs[RNN] = encoder.birnn.Config()
    self.subcfgs[TXT_EMBED] = framework.impl.encoder.pca.Config()
    self.subcfgs[FT_EMBED] = framework.impl.encoder.pca.Config()

    self.max_words_in_caption = 30
    self.pool = 'mean'

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = .1
    self.alpha = 0.5
    self.num_neg = 1
    self.beta = 0.5
    self.loss = 'norm'

  def _assert(self):
    assert self.max_words_in_caption == self.subcfgs[RNN].num_step
    assert self.subcfgs[WE].dim_embed == self.subcfgs[RNN].subcfgs[CELL].dim_input
    assert self.subcfgs[TXT_EMBED].dim_ft == self.subcfgs[RNN].subcfgs[CELL].dim_hidden + self.subcfgs[RNN].subcfgs[RCELL].dim_hidden
    assert self.subcfgs[FT_EMBED].dim_ft == self.dim_ft


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
  cfg.dim_ft = kwargs['dim_ft']
  cfg.dim_joint_embed = kwargs['dim_joint_embed']
  cfg.beta = kwargs['beta']

  cfg.max_words_in_caption = kwargs['max_words_in_caption']
  cfg.pool = kwargs['pool']
  cfg.loss = kwargs['loss']

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

  txt_embed_cfg = cfg.subcfgs[TXT_EMBED]
  txt_embed_cfg.dim_ft = 2*kwargs['cell_dim_hidden']
  txt_embed_cfg.dim_output = kwargs['dim_joint_embed']

  ft_embed_cfg = cfg.subcfgs[FT_EMBED]
  ft_embed_cfg.dim_ft = kwargs['dim_ft']
  ft_embed_cfg.dim_output = kwargs['dim_joint_embed']

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
    REGULAR = 'regularization'

  def _set_submods(self):
    return {
      WE: encoder.word.Encoder(self._config.subcfgs[WE]),
      RNN: encoder.birnn.Encoder(self._config.subcfgs[RNN]),
      TXT_EMBED: framework.impl.encoder.pca.Encoder1D(self._config.subcfgs[TXT_EMBED]),
      FT_EMBED: framework.impl.encoder.pca.Encoder(self._config.subcfgs[FT_EMBED]),
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
    # self.scale = tf.contrib.framework.model_variable('scale',
    #   shape=(), dtype=tf.float32, initializer=tf.constant_initializer(1.))
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

    txt_pca = self.submods[TXT_EMBED]
    out_ops = txt_pca.get_out_ops_in_mode({
      txt_pca.InKey.FT: out_ops[rnn.OutKey.OUTPUT],
    }, mode)
    caption_embed = out_ops[txt_pca.OutKey.EMBED]

    ft_pca = self.submods[FT_EMBED]
    out_ops = ft_pca.get_out_ops_in_mode({
      ft_pca.InKey.FT: in_ops[self.InKey.FT]
    }, mode)
    ft_embed = out_ops[ft_pca.OutKey.EMBED]

    with tf.variable_scope(self.name_scope):
      mask = in_ops[self.InKey.CAPTION_MASK]
      mask = tf.expand_dims(tf.to_float(mask), 2)
      if self._config.pool == 'mean':
        caption_embed = tf.reduce_sum(caption_embed*mask, 1) / tf.reduce_sum(mask, 1)
      else:
        base = tf.reduce_min(caption_embed, 1)
        caption_embed -= tf.expand_dims(base, 1)
        caption_embed = tf.reduce_max(caption_embed * mask, 1)
        caption_embed += base

      if self._config.loss == 'norm':
        caption_norm = tf.norm(caption_embed, axis=-1)
        ft_norm = tf.norm(ft_embed, axis=-1)
        regularization = tf.abs(ft_norm - 0.01*tf.ones_like(ft_norm)) + tf.abs(caption_norm - 0.01*tf.ones_like(caption_norm))

      # unit ball
      caption_embed = tf.clip_by_norm(caption_embed, 1.0-1e-5, 1)
      self.op2monitor['caption_embed_norm'] = tf.reduce_mean(tf.norm(caption_embed, axis=-1))
      caption_embed_poincare = framework.util.expanded_op.poincareball_gradient(caption_embed)

      # unit ball
      ft_embed = tf.clip_by_norm(ft_embed, 1.0-1e-5, 1)
      self.op2monitor['ft_embed_norm'] = tf.reduce_mean(tf.norm(ft_embed, axis=-1))
      ft_embed_poincare = framework.util.expanded_op.poincareball_gradient(ft_embed)

      if self._config.loss != 'norm':
        regularization = tf.norm(ft_embed, axis=-1) + tf.norm(caption_embed, axis=-1)

    def trn(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        pos_ft_embed = ft_embed[:-self._config.num_neg]
        pos_caption_embed = caption_embed[:-self._config.num_neg]
        neg_ft_embed = ft_embed[-self._config.num_neg:]
        neg_caption_embed = caption_embed[-self._config.num_neg:]

        pos_dist = tf.square(tf.norm(pos_ft_embed - pos_caption_embed, axis=-1))
        pos_dist /= (1. - tf.square(tf.norm(pos_ft_embed, axis=-1))) * (1. - tf.square(tf.norm(pos_caption_embed, axis=-1)))
        pos_dist = 1 + 2 * pos_dist
        pos_dist = tf.acosh(pos_dist)
        pos_sim = -pos_dist
        # pos_sim *= self.scale

        neg_caption_dist = tf.square(tf.norm(tf.expand_dims(pos_ft_embed, 1) - tf.expand_dims(neg_caption_embed, 0), axis=-1))
        neg_caption_dist /= 1. - tf.square(tf.norm(tf.expand_dims(pos_ft_embed, 1), axis=-1))
        neg_caption_dist /= 1. - tf.square(tf.norm(tf.expand_dims(neg_caption_embed, 0), axis=-1))
        neg_caption_dist = 1 + 2 * neg_caption_dist
        neg_caption_dist = tf.acosh(neg_caption_dist)
        if self._config.loss == 'lifted':
          neg_caption_sim = -neg_caption_dist
          # neg_caption_sim *= self.scale
          neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1) / 100.
        else:
          neg_caption_sim = -neg_caption_dist
          # neg_caption_sim *= self.scale
          neg_caption_sim = tf.concat([neg_caption_sim, tf.expand_dims(pos_sim, 1)], 1)
          neg_caption_sim = tf.reduce_logsumexp(neg_caption_sim, 1)

        neg_ft_dist = tf.square(tf.norm(tf.expand_dims(pos_caption_embed, 1) - tf.expand_dims(neg_ft_embed, 0), axis=-1))
        neg_ft_dist /= 1. - tf.square(tf.norm(tf.expand_dims(pos_caption_embed, 1), axis=-1))
        neg_ft_dist /= 1. - tf.square(tf.norm(tf.expand_dims(neg_ft_embed, 0), axis=-1))
        neg_ft_dist = 1 + 2 * neg_ft_dist
        neg_ft_dist = tf.acosh(neg_ft_dist)
        if self._config.loss == 'lifted':
          neg_ft_sim = -neg_ft_dist
          # neg_ft_sim *= self.scale
          neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1) / 100.
        else:
          neg_ft_sim = -neg_ft_dist
          # neg_ft_sim *= self.scale
          neg_ft_sim = tf.concat([neg_ft_sim, tf.expand_dims(pos_sim, 1)], 1)
          neg_ft_sim = tf.reduce_logsumexp(neg_ft_sim, 1)

      return pos_sim, neg_caption_sim, neg_ft_sim

    def tst(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        dist = tf.square(tf.norm(tf.expand_dims(ft_embed, 1) - tf.expand_dims(caption_embed, 0), axis=-1))
        dist /= 1. - tf.square(tf.norm(tf.expand_dims(ft_embed, 1), axis=-1))
        dist /= 1. - tf.square(tf.norm(tf.expand_dims(caption_embed, 0), axis=-1))
        dist = 1 + 2 * dist
        dist = tf.acosh(dist)
        sim = -dist
      return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      if self._config.loss == 'norm':
        return {
          self.OutKey.REGULAR: regularization,
        }
      else:
        pos_sim, neg_caption_sim, neg_ft_sim = trn(ft_embed_poincare, caption_embed_poincare)
        sim = tst(ft_embed, caption_embed)
        return {
          self.OutKey.SIM: sim,
          self.OutKey.P_SIM: pos_sim,
          self.OutKey.NF_SIM: neg_ft_sim,
          self.OutKey.NC_SIM: neg_caption_sim,
          self.OutKey.REGULAR: regularization,
        }
    else:
      sim = tst(ft_embed, caption_embed)
      return {
        self.OutKey.SIM: sim,
      }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      if self._config.loss == 'norm':
        regularization = self._outputs[self.OutKey.REGULAR]
        self.op2monitor['loss'] = tf.reduce_mean(regularization)
        loss = tf.reduce_mean(regularization)
      else:
        pos_sim = self._outputs[self.OutKey.P_SIM]
        neg_caption_sim = self._outputs[self.OutKey.NC_SIM]
        neg_ft_sim = self._outputs[self.OutKey.NF_SIM]
        regularization = self._outputs[self.OutKey.REGULAR]
        self.op2monitor['pos_sim'] = tf.reduce_mean(pos_sim)
        self.op2monitor['neg_caption_sim'] = tf.reduce_mean(neg_caption_sim)
        self.op2monitor['neg_ft_sim'] = tf.reduce_mean(neg_ft_sim)
        self.op2monitor['regularization'] = tf.reduce_mean(regularization)
        # self.op2monitor['scale'] = self.scale

        if self._config.loss == 'lifted':
          # contrast_caption_loss = self.scale*neg_caption_sim + self._config.margin - self.scale*pos_sim
          contrast_caption_loss = neg_caption_sim + self._config.margin - pos_sim
          contrast_caption_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_caption_loss))
        else:
          # contrast_caption_loss = self.scale*(neg_caption_sim - pos_sim)
          contrast_caption_loss = neg_caption_sim - pos_sim

        if self._config.loss == 'lifted':
          # contrast_ft_loss = self.scale*neg_ft_sim + self._config.margin - self.scale*pos_sim
          contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
          contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))
        else:
          # contrast_ft_loss = self.scale*(neg_ft_sim - pos_sim)
          contrast_ft_loss = neg_ft_sim - pos_sim

        loss = self._config.alpha * contrast_caption_loss + (1.0 - self._config.alpha) * contrast_ft_loss
        loss = tf.reduce_mean(loss)
        loss += self._config.beta * tf.reduce_mean(regularization)
        self.op2monitor['loss'] = loss
    return loss

  def op_in_val(self, **kwargs):
    if self._config.loss == 'norm':
      return {
        self.OutKey.REGULAR: self._outputs[self.OutKey.REGULAR],
      }
    else:
      return {
        self.OutKey.SIM: self._outputs[self.OutKey.SIM],
      }

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.SIM: self._outputs[self.OutKey.SIM],
    }


PathCfg = trntst_util.PathCfg
TrnTst = trntst_util.TrnTst
NormTrnTst = trntst_util.NormTrnTst


TrnReader = trntst_util.TrnReader
ValReader = trntst_util.ValReader
TstReader = trntst_util.TstReader

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
    self.margin = 0.1
    self.alpha = 0.5
    self.num_neg = 1
    self.l2norm = True

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

  for cell in [CELL, RCELL]:
    cell_cfg = rnn_cfg.subcfgs[cell]
    cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
    cell_cfg.dim_input = 300
    cell_cfg.keepout_prob = 0.5
    cell_cfg.keepin_prob = 0.5

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'rnnve.Model'

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
      self.ft_pca_W = tf.contrib.framework.model_variable('ft_pca_W',
        shape=(self._config.dim_ft, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.ft_pca_B = tf.contrib.framework.model_variable('ft_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.ft_pca_W)
      self._weights.append(self.ft_pca_B)

      self.caption_pca_Ws = []
      self.caption_pca_Bs = []
      for l in range(2):
        dim_input = self._config.subcfgs[WE].dim_embed if l == 0 else self._config.dim_joint_embed
        caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W_%d'%l,
          shape=(dim_input, self._config.dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        caption_pca_B = tf.contrib.framework.model_variable('caption_pca_B_%d'%l,
          shape=(self._config.dim_joint_embed,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._weights.append(caption_pca_W)
        self._weights.append(caption_pca_B)
        self.caption_pca_Ws.append(caption_pca_W)
        self.caption_pca_Bs.append(caption_pca_B)

      dim_hidden = self._config.subcfgs[RNN].subcfgs[CELL].dim_hidden
      self.caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W',
        shape=(2*dim_hidden, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.caption_pca_B = tf.contrib.framework.model_variable('caption_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.caption_pca_W)
      self._weights.append(self.caption_pca_B)

      self.word_att_Ws = []
      self.word_att_Bs = []
      for l in range(2):
        word_att_W = tf.contrib.framework.model_variable('word_att_W_%d'%l,
          shape=(self._config.dim_joint_embed, self._config.dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        word_att_B = tf.contrib.framework.model_variable('word_att_B_%d'%l,
          shape=(self._config.dim_joint_embed,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._weights.append(word_att_W)
        self._weights.append(word_att_B)
        self.word_att_Ws.append(word_att_W)
        self.word_att_Bs.append(word_att_B)

      self.ft_att_Ws = []
      self.ft_att_Bs = []
      for l in range(2):
        dim_input = self._config.dim_ft if l == 0 else self._config.dim_joint_embed
        ft_att_W = tf.contrib.framework.model_variable('ft_att_W_%d'%l,
          shape=(dim_input, self._config.dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        ft_att_B = tf.contrib.framework.model_variable('ft_att_B_%d'%l,
          shape=(self._config.dim_joint_embed,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._weights.append(ft_att_W)
        self._weights.append(ft_att_B)
        self.ft_att_Ws.append(ft_att_W)
        self.ft_att_Bs.append(ft_att_B)

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
      if self._config.l2norm:
        caption_embed = tf.nn.l2_normalize(caption_embed, 1)

      ft_embed = tf.nn.xw_plus_b(in_ops[self.InKey.FT], self.ft_pca_W, self.ft_pca_B)
      ft_embed = tf.nn.tanh(ft_embed)
      if self._config.l2norm:
        ft_embed = tf.nn.l2_normalize(ft_embed, 1)

    def trn(wvecs, fts, caption_embed, ft_embed, mask, is_trn):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(wvecs)[0]
        num_pos = batch_size - self._config.num_neg
        num_neg = self._config.num_neg
        num_word = self._config.max_words_in_caption
        dim_word = self._config.subcfgs[WE].dim_embed
        dim_ft = self._config.dim_ft
        dim_embed = self._config.dim_joint_embed

        # embed
        pos_fts = fts[:num_pos]
        neg_fts = fts[num_pos:]

        wvecs = tf.reshape(wvecs, (-1, dim_word))
        wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[0], self.caption_pca_Bs[0])
        wvecs = tf.nn.relu(wvecs)
        wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[1], self.caption_pca_Bs[1])
        wvecs = tf.reshape(wvecs, (-1, num_word, dim_embed))
        wvecs = tf.nn.l2_normalize(wvecs, -1)
        pos_wvecs = wvecs[:num_pos]
        neg_wvecs = wvecs[num_pos:]

        mask = tf.to_float(mask)
        pos_mask = mask[:num_pos]
        neg_mask = mask[num_pos:]

        alpha = tf.nn.xw_plus_b(tf.reshape(wvecs, (-1, dim_embed)), self.word_att_Ws[0], self.word_att_Bs[0])
        alpha = tf.nn.relu(alpha) # (None, num_word, dim_embed)
        alpha = tf.nn.xw_plus_b(alpha, self.word_att_Ws[1], self.word_att_Bs[1])
        alpha = tf.reshape(alpha, (-1, num_word, dim_embed))
        beta = tf.nn.xw_plus_b(fts, self.ft_att_Ws[0], self.ft_att_Bs[0])
        beta = tf.nn.relu(beta)
        beta = tf.nn.xw_plus_b(beta, self.ft_att_Ws[1], self.ft_att_Bs[1])
        beta = tf.expand_dims(beta, 1) # (None, 1, dim_embed)
        pos_alpha = alpha[:num_pos]
        neg_alpha = alpha[num_pos:]
        pos_beta = beta[:num_pos]
        neg_beta = beta[num_pos:]

        def calc_pos_attwv(pos_fts, pos_wvecs, pos_alpha, pos_beta, pos_mask):
          print pos_alpha.get_shape()
          print pos_beta.get_shape()
          # attend
          att = tf.matmul(pos_alpha, pos_beta, transpose_b=True) # (num_pos, num_word, 1)
          att = att[:, :, 0] # (num_pos, num_word)
          att = tf.nn.softmax(att, 1)
          att *= pos_mask
          att /= tf.reduce_sum(att, 1, True)
          print att.get_shape()
          wvec_bar = tf.reduce_sum(pos_wvecs * tf.expand_dims(att, 2), 1) # (num_pos, dim_embed)
          wvec_bar = tf.nn.l2_normalize(wvec_bar, -1)

          return wvec_bar

        def calc_neg_word_attwv(pos_fts, neg_wvecs, neg_alpha, pos_beta, neg_mask):
          # attend
          neg_alpha = tf.reshape(neg_alpha, (-1, dim_embed)) # (num_neg*num_word, dim_embed)
          pos_beta = tf.reshape(pos_beta, (-1, dim_embed)) # (num_pos, dim_embed)
          att = tf.matmul(neg_alpha, pos_beta, transpose_b=True)
          att = tf.reshape(att, (num_neg, num_word, num_pos)) # (num_neg, num_word, num_pos)
          att = tf.nn.softmax(att, 1)
          att *= tf.expand_dims(neg_mask, 2)
          att /= tf.reduce_sum(att, 1, True)
          wvecs_bar = tf.reduce_sum(
            tf.expand_dims(neg_wvecs, 2) * tf.expand_dims(att, 3), 1) # (num_neg, num_pos, dim_embed)
          wvecs_bar = tf.nn.l2_normalize(wvecs_bar, -1)

          return wvecs_bar

        def calc_neg_ft_sim(neg_fts, pos_wvecs, pos_alpha, neg_beta, pos_mask):
          # attend
          pos_alpha = tf.reshape(pos_alpha, (-1, dim_embed)) # (num_pos*num_word, dim_embed)
          neg_beta = tf.reshape(neg_beta, (-1, dim_embed)) # (num_neg, dim_embed)
          att = tf.matmul(neg_beta, pos_alpha, transpose_b=True) # (num_eng, num_pos*num_word)
          att = tf.reshape(att, (num_neg, num_pos, num_word)) # (num_neg, num_pos, num_word)
          att = tf.nn.softmax(att, 2)
          att *= tf.expand_dims(pos_mask, 0)
          att /= tf.reduce_sum(att, 2, True)
          wvecs_bar = tf.reduce_sum(
            tf.expand_dims(pos_wvecs, 0) * tf.expand_dims(att, 3), 2) # (num_neg, num_pos, dim_embed)
          wvecs_bar = tf.nn.l2_normalize(wvecs_bar, -1)

          return wvecs_bar

        pos_attwv = calc_pos_attwv(pos_fts, pos_wvecs, pos_alpha, pos_beta, pos_mask)
        neg_word_attwv = calc_neg_word_attwv(pos_fts, neg_wvecs, neg_alpha, pos_beta, neg_mask)
        neg_ft_attwv = calc_neg_ft_sim(neg_fts, pos_wvecs, pos_alpha, neg_beta, pos_mask)

        pos_ft_embed = ft_embed[:-self._config.num_neg]
        pos_caption_embed = caption_embed[:-self._config.num_neg]
        neg_ft_embed = ft_embed[-self._config.num_neg:]
        neg_caption_embed = caption_embed[-self._config.num_neg:]

        pos_sim = tf.reduce_sum(pos_ft_embed * pos_caption_embed, 1) # (num_pos,)
        pos_sim += tf.reduce_sum(pos_ft_embed * pos_attwv, 1)
        neg_caption_sim = tf.matmul(neg_caption_embed, pos_ft_embed, transpose_b=True)
        neg_caption_sim += tf.reduce_sum(tf.expand_dims(pos_ft_embed, 0) * neg_word_attwv, 2) # (num_neg, num_pos)
        neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 0) / 100.
        neg_ft_sim = tf.matmul(neg_ft_embed, pos_caption_embed, transpose_b=True)
        neg_ft_sim += tf.reduce_sum(tf.expand_dims(neg_ft_embed, 1) * pos_caption_embed, 2) # (num_neg, num_pos)
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 0) / 100.

        return pos_sim, neg_caption_sim, neg_ft_sim

    def tst(wvecs, fts, caption_embed, ft_embed, mask, is_trn):
      with tf.variable_scope(self.name_scope):
        num_ft = tf.shape(fts)[0]
        num_caption = tf.shape(wvecs)[0]
        num_word = self._config.max_words_in_caption
        dim_word = self._config.subcfgs[WE].dim_embed
        dim_ft = self._config.dim_ft
        dim_embed = self._config.dim_joint_embed
        mask = tf.to_float(mask)

        wvecs = tf.reshape(wvecs, (-1, dim_word))
        wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[0], self.caption_pca_Bs[0])
        wvecs = tf.nn.relu(wvecs)
        wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[1], self.caption_pca_Bs[1])
        wvecs = tf.reshape(wvecs, (-1, num_word, dim_embed))
        wvecs = tf.nn.l2_normalize(wvecs, -1)

        # attend
        alpha = tf.nn.xw_plus_b(tf.reshape(wvecs, (-1, dim_embed)), self.word_att_Ws[0], self.word_att_Bs[0])
        alpha = tf.nn.relu(alpha) # (num_caption*num_word, dim_embed)
        alpha = tf.nn.xw_plus_b(alpha, self.word_att_Ws[1], self.word_att_Bs[1])
        beta = tf.nn.xw_plus_b(fts, self.ft_att_Ws[0], self.ft_att_Bs[0])
        beta = tf.nn.relu(beta)
        beta = tf.nn.xw_plus_b(beta, self.ft_att_Ws[1], self.ft_att_Bs[1])

        att = tf.matmul(beta, alpha, transpose_b=True) # (num_ft, num_caption*num_word)
        att = tf.reshape(att, (num_ft, num_caption, num_word))
        att = tf.nn.softmax(att, 2)
        att *= tf.expand_dims(mask, 0)
        att /= tf.reduce_sum(att, 2, True)
        wvecs_bar = tf.reduce_sum(
          tf.expand_dims(wvecs, 0) * tf.expand_dims(att, 3), 2) # (num_ft, num_caption, dim_embed)
        wvecs_bar = tf.nn.l2_normalize(wvecs_bar, -1)

        sim = tf.matmul(ft_embed, caption_embed, transpose_b=True)
        sim += tf.reduce_sum(tf.expand_dims(ft_embed, 1) * wvecs_bar, 2)

        return sim

    is_trn = in_ops[self.InKey.IS_TRN]
    fts = in_ops[self.InKey.FT]
    if mode == framework.model.module.Mode.TRN_VAL:
      pos_sim, neg_caption_sim, neg_ft_sim = trn(wvecs, fts, caption_embed, ft_embed, mask, is_trn)
      sim = tst(wvecs, fts, caption_embed, ft_embed, mask, is_trn)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_caption_sim,
      }
    else:
      sim = tst(wvecs, fts, caption_embed, ft_embed, mask, is_trn)
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

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
import trntst_util

WE = 'word'


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[WE] = encoder.word.Config()

    self.max_words_in_caption = 30

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = 0.1
    self.alpha = 0.5
    self.num_neg = 1
    self.num_track = 11


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
  cfg.dim_ft = kwargs['dim_ft']
  cfg.dim_joint_embed = kwargs['dim_joint_embed']
  cfg.num_track = kwargs['num_track']

  cfg.max_words_in_caption = kwargs['max_words_in_caption']

  we_cfg = cfg.subcfgs[WE]
  we_cfg.lr_mult = 1e-2

  return cfg


class Model(framework.model.module.AbstractModel):
  """
  A Decomposable Attention Model for Natural Language Inference
  https://aclweb.org/anthology/D16-1244
  """
  name_scope = 'aca.Model'

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
    return {
      WE: encoder.word.Encoder(self._config.subcfgs[WE]),
    }

  def _add_input_in_mode(self, mode):
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, self._config.num_track, self._config.dim_ft), name='fts')
      ft_masks = tf.placeholder(
        tf.float32, shape=(None, self._config.num_track), name='ft_masks')
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
      self.ft_pca_Ws = []
      self.ft_pca_Bs = []
      for l in range(2):
        dim_input = self._config.dim_ft if l == 0 else self._config.dim_joint_embed
        ft_pca_W = tf.contrib.framework.model_variable('ft_pca_W_%d'%l,
          shape=(dim_input, self._config.dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        ft_pca_B = tf.contrib.framework.model_variable('ft_pca_B_%d'%l,
          shape=(self._config.dim_joint_embed,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._weights.append(ft_pca_W)
        self._weights.append(ft_pca_B)
        self.ft_pca_Ws.append(ft_pca_W)
        self.ft_pca_Bs.append(ft_pca_B)

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
        ft_att_W = tf.contrib.framework.model_variable('ft_att_W_%d'%l,
          shape=(self._config.dim_joint_embed, self._config.dim_joint_embed), dtype=tf.float32,
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
    wvecs = out_ops[encoder.OutKey.EMBED]
    fts = in_ops[self.InKey.FT]
    ft_mask = in_ops[self.InKey.FT_MASK]
    word_mask = in_ops[self.InKey.CAPTION_MASK]
    is_trn = in_ops[self.InKey.IS_TRN]

    def trn(wvecs, fts, word_mask, ft_mask, is_trn):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(wvecs)[0]
        num_pos = batch_size - self._config.num_neg
        num_neg = self._config.num_neg
        num_word = self._config.max_words_in_caption
        dim_word = self._config.subcfgs[WE].dim_embed
        dim_ft = self._config.dim_ft
        num_ft = self._config.num_track
        dim_embed = self._config.dim_joint_embed

        # embed
        fts = tf.reshape(fts, (-1, dim_ft))
        fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[0], self.ft_pca_Bs[0])
        fts = tf.nn.relu(fts)
        fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[1], self.ft_pca_Bs[1])
        fts = tf.reshape(fts, (-1, num_ft, dim_embed))
        fts = tf.nn.l2_normalize(fts, -1)
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

        word_mask = tf.to_float(word_mask)
        pos_word_mask = word_mask[:num_pos]
        neg_word_mask = word_mask[num_pos:]
        pos_ft_mask = ft_mask[:num_pos]
        neg_ft_mask = ft_mask[num_pos:]

        alpha = tf.nn.xw_plus_b(tf.reshape(wvecs, (-1, dim_embed)), self.word_att_Ws[0], self.word_att_Bs[0])
        alpha = tf.nn.relu(alpha)
        alpha = tf.nn.xw_plus_b(alpha, self.word_att_Ws[1], self.word_att_Bs[1])
        alpha = tf.reshape(alpha, (-1, num_word, dim_embed))
        beta = tf.nn.xw_plus_b(tf.reshape(fts, (-1, dim_embed)), self.ft_att_Ws[0], self.ft_att_Bs[0])
        beta = tf.nn.relu(beta)
        beta = tf.nn.xw_plus_b(beta, self.ft_att_Ws[1], self.ft_att_Bs[1])
        beta = tf.reshape(beta, (-1, num_ft, dim_embed))
        pos_alpha = alpha[:num_pos]
        neg_alpha = alpha[num_pos:]
        pos_beta = beta[:num_pos]
        neg_beta = beta[num_pos:]

        def calc_pos_sim(pos_fts, pos_wvecs, pos_alpha, pos_beta, pos_word_mask, pos_ft_mask):
          # attend
          att = tf.matmul(pos_alpha, pos_beta, transpose_b=True) # (num_pos, num_word, num_ft)

          word_att = tf.nn.softmax(att, 1)
          word_att *= tf.expand_dims(pos_word_mask, 2)
          word_att /= tf.reduce_sum(word_att, 1, True)
          wvec_bar = tf.reduce_sum(tf.expand_dims(pos_wvecs, 2) * tf.expand_dims(word_att, 3), 1) # (num_pos, num_ft, dim_embed)
          # wvec_bar = tf.nn.l2_normalize(wvec_bar, -1)
          wvecs_bar /= tf.reshape(tf.reduce_sum(word_att, [1, 2]), (num_pos, 1, 1))

          ft_att = tf.nn.softmax(att, 2)
          ft_att *= tf.expand_dims(pos_ft_mask, 1)
          ft_att /= tf.reduce_sum(ft_att, 2, True)
          ft_bar = tf.reduce_sum(tf.expand_dims(pos_fts, 1) * tf.expand_dims(ft_att, 3), 2) # (num_pos, num_word, dim_embed)
          # ft_bar = tf.nn.l2_normalize(ft_bar, -1)
          ft_bar /= tf.reshape(tf.reduce_sum(ft_att, [1, 2]), (num_pos, 1, 1))

          # compare
          wvec_compare = tf.reduce_sum(ft_bar * pos_wvecs, -1) # (num_pos, num_word)
          ft_compare = tf.reduce_sum(wvec_bar * pos_fts, -1) # (num_pos, num_ft)

          # aggregate
          norm = tf.reduce_sum(pos_word_mask, 1) * tf.reduce_sum(pos_ft_mask, 1)
          pos_sim = tf.reduce_sum(wvec_compare * pos_word_mask, -1) / norm
          pos_sim += tf.reduce_sum(ft_compare * pos_ft_mask, -1) / norm
          pos_sim /= 2.

          return pos_sim

        def calc_neg_word_sim(pos_fts, neg_wvecs, neg_alpha, pos_beta, neg_word_mask, pos_ft_mask):
          # attend
          neg_alpha = tf.reshape(neg_alpha, (-1, dim_embed)) # (num_neg*num_word, dim_embed)
          pos_beta = tf.reshape(pos_beta, (-1, dim_embed)) # (num_pos*num_ft, dim_embed)
          att = tf.matmul(neg_alpha, pos_beta, transpose_b=True)
          att = tf.reshape(att, (num_neg, num_word, num_pos, num_ft))

          word_att = tf.nn.softmax(att, 1)
          word_att *= tf.reshape(neg_word_mask, (num_neg, num_word, 1, 1))
          word_att /= tf.reduce_sum(word_att, 1, True)
          wvecs_bar = tf.reduce_sum(
            tf.reshape(neg_wvecs, (num_neg, num_word, 1, 1, dim_embed)) * tf.expand_dims(word_att, 4), 1) # (num_neg, num_pos, num_ft, dim_embed)
          # wvecs_bar = tf.nn.l2_normalize(wvecs_bar, -1)
          wvecs_bar /= tf.reshape(tf.reduce_sum(word_att, [1, 3]), (num_neg, num_pos, 1, 1))

          ft_att = tf.nn.softmax(att, 3)
          ft_att *= tf.reshape(pos_ft_mask, (1, 1, num_pos, num_ft))
          ft_att /= tf.reduce_sum(ft_att, 3, True)
          ft_bar = tf.reduce_sum(
            tf.reshape(pos_fts, (1, 1, num_pos, num_ft, dim_embed)) * tf.expand_dims(ft_att, 4), 3) # (num_neg, num_word, num_pos, dim_embed)
          # ft_bar = tf.nn.l2_normalize(ft_bar, -1)
          ft_bar /= tf.reshape(tf.reduce_sum(ft_att, [1, 3]), (num_neg, 1, num_pos, 1))

          # compare
          wvec_compare = tf.reduce_sum(ft_bar * tf.expand_dims(neg_wvecs, 2), -1) # (num_neg, num_word, num_pos)
          ft_compare = tf.reduce_sum(wvecs_bar * tf.expand_dims(pos_fts, 0), -1) # (num_neg, num_pos, num_ft)

          # aggregate
          norm = tf.expand_dims(tf.reduce_sum(neg_word_mask, 1), 1) * tf.expand_dims(tf.reduce_sum(pos_ft_mask, 1), 0)
          neg_sim = tf.reduce_sum(wvec_compare * tf.expand_dims(neg_word_mask, 2), 1) / norm
          neg_sim += tf.reduce_sum(ft_compare * tf.expand_dims(pos_ft_mask, 0), 2) / norm
          neg_sim /= 2.

          return neg_sim

        def calc_neg_ft_sim(neg_fts, pos_wvecs, pos_alpha, neg_beta, pos_word_mask, neg_ft_mask):
          # attend
          pos_alpha = tf.reshape(pos_alpha, (-1, dim_embed)) # (num_pos*num_word, dim_embed)
          neg_beta = tf.reshape(neg_beta, (-1, dim_embed)) # (num_neg*num_ft, dim_embed)
          att = tf.matmul(neg_beta, pos_alpha, transpose_b=True)
          att = tf.reshape(att, (num_neg, num_ft, num_pos, num_word))

          word_att = tf.nn.softmax(att, 3)
          word_att *= tf.reshape(pos_word_mask, (1, 1, num_pos, num_word))
          word_att /= tf.reduce_sum(word_att, 3, True)
          wvecs_bar = tf.reduce_sum(
            tf.reshape(pos_wvecs, (1, 1, num_pos, num_word, dim_embed)) * tf.expand_dims(word_att, 4), 3) # (num_neg, num_ft, num_pos, dim_embed)
          # wvecs_bar = tf.nn.l2_normalize(wvecs_bar, -1)
          wvecs_bar /= tf.reshape(tf.reduce_sum(word_att, [1, 3]), (num_neg, 1, num_pos, 1))

          ft_att = tf.nn.softmax(att, 1)
          ft_att *= tf.reshape(neg_ft_mask, (num_neg, num_ft, 1, 1))
          ft_att /= tf.reduce_sum(ft_att, 1, True)
          ft_bar = tf.reduce_sum(
            tf.reshape(neg_fts, (num_neg, num_ft, 1, 1, dim_embed)) * tf.expand_dims(ft_att, 4), 1) # (num_neg, num_pos, num_word, dim_embed)
          # ft_bar = tf.nn.l2_normalize(ft_bar, -1)
          ft_bar /= tf.reshape(tf.reduce_sum(ft_att, [1, 3]), (num_neg, num_pos, 1, 1))

          # compare
          wvec_compare = tf.reduce_sum(ft_bar * tf.expand_dims(pos_wvecs, 0), -1) # (num_neg, num_pos, num_word)
          ft_compare = tf.reduce_sum(wvecs_bar * tf.expand_dims(neg_fts, 2), -1) # (num_neg, num_ft, num_pos)

          # aggregate
          norm = tf.expand_dims(tf.reduce_sum(pos_word_mask, 1), 0) * tf.expand_dims(tf.reduce_sum(neg_ft_mask, 1), 1)
          neg_sim = tf.reduce_sum(wvec_compare * tf.expand_dims(pos_word_mask, 0), 2) / norm
          neg_sim += tf.reduce_sum(ft_compare * tf.expand_dims(neg_ft_mask, 2), 1) / norm
          neg_sim /= 2.

          return neg_sim

        pos_sim = calc_pos_sim(pos_fts, pos_wvecs, pos_alpha, pos_beta, pos_word_mask, pos_ft_mask)
        neg_word_sim = calc_neg_word_sim(pos_fts, neg_wvecs, neg_alpha, pos_beta, neg_word_mask, pos_ft_mask)
        neg_ft_sim = calc_neg_ft_sim(neg_fts, pos_wvecs, pos_alpha, neg_beta, pos_word_mask, neg_ft_mask)
        neg_word_sim = tf.reduce_logsumexp(100.*neg_word_sim, 0) / 100. # (num_pos,)
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 0) / 100.

        return pos_sim, neg_word_sim, neg_ft_sim

    def tst(wvecs, fts, word_mask, ft_mask, is_trn):
      with tf.variable_scope(self.name_scope):
        num_ft = tf.shape(fts)[0]
        num_caption = tf.shape(wvecs)[0]
        num_word = self._config.max_words_in_caption
        dim_word = self._config.subcfgs[WE].dim_embed
        dim_ft = self._config.dim_ft
        num_track = self._config.num_track
        dim_embed = self._config.dim_joint_embed
        word_mask = tf.to_float(word_mask)

        # embed
        fts = tf.reshape(fts, (-1, dim_ft))
        fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[0], self.ft_pca_Bs[0])
        fts = tf.nn.relu(fts)
        fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[1], self.ft_pca_Bs[1])
        fts = tf.reshape(fts, (-1, num_track, dim_embed))
        fts = tf.nn.l2_normalize(fts, -1)

        wvecs = tf.reshape(wvecs, (-1, dim_word))
        wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[0], self.caption_pca_Bs[0])
        wvecs = tf.nn.relu(wvecs)
        wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[1], self.caption_pca_Bs[1])
        wvecs = tf.reshape(wvecs, (-1, num_word, dim_embed))
        wvecs = tf.nn.l2_normalize(wvecs, -1)

        # attend
        alpha = tf.nn.xw_plus_b(tf.reshape(wvecs, (-1, dim_embed)), self.word_att_Ws[0], self.word_att_Bs[0])
        alpha = tf.nn.relu(alpha) # (num_caption * num_word, dim_embed)
        alpha = tf.nn.xw_plus_b(alpha, self.word_att_Ws[1], self.word_att_Bs[1])
        beta = tf.nn.xw_plus_b(tf.reshape(fts, (-1, dim_embed)), self.ft_att_Ws[0], self.ft_att_Bs[0])
        beta = tf.nn.relu(beta) # (num_ft * num_track, dim_embed)
        beta = tf.nn.xw_plus_b(beta, self.ft_att_Ws[1], self.ft_att_Bs[1])

        att = tf.matmul(beta, alpha, transpose_b=True) # (num_ft*num_track, num_caption*num_word)
        att = tf.reshape(att, (num_ft, num_track, num_caption, num_word))

        word_att = tf.nn.softmax(att, 3)
        word_att *= tf.reshape(word_mask, (1, 1, num_caption, num_word))
        word_att /= tf.reduce_sum(word_att, 3, True)
        wvecs_bar = tf.reduce_sum(
          tf.reshape(wvecs, (1, 1, num_caption, num_word, dim_embed)) * tf.expand_dims(word_att, 4), 3)
        # wvecs_bar = tf.nn.l2_normalize(wvecs_bar, -1) # (num_ft, num_track, num_caption, dim_embed)
        wvecs_bar /= tf.reshape(tf.reduce_sum(word_att, [1, 3]), (num_ft, 1, num_caption, 1))

        ft_att = tf.nn.softmax(att, 1)
        ft_att *= tf.reshape(ft_mask, (num_ft, num_track, 1, 1))
        ft_att /= tf.reduce_sum(ft_att, 1, True)
        ft_bar = tf.reduce_sum(
          tf.reshape(fts, (num_ft, num_track, 1, 1, dim_embed)) * tf.expand_dims(ft_att, 4), 1)
        # ft_bar = tf.nn.l2_normalize(ft_bar, -1) # (num_ft, num_caption, num_word, dim_embed)
        ft_bar /= tf.reshape(tf.reduce_sum(ft_att, [1, 3]), (num_ft, num_caption, 1, 1))

        # compare
        wvec_compare = tf.reduce_sum(ft_bar * tf.expand_dims(wvecs, 0), -1) # (num_ft, num_caption, num_word)
        ft_compare = tf.reduce_sum(wvecs_bar * tf.expand_dims(fts, 2), -1) # (num_ft, num_track, num_caption)

        # aggregate
        norm = tf.expand_dims(tf.reduce_sum(word_mask, 1), 0) * tf.expand_dims(tf.reduce_sum(ft_mask, 1), 1)
        sim = tf.reduce_sum(wvec_compare * tf.expand_dims(word_mask, 0), 2) / norm
        sim += tf.reduce_sum(ft_compare * tf.expand_dims(ft_mask, 2), 1) / norm
        sim /= 2.

        return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      pos_sim, neg_word_sim, neg_ft_sim = trn(wvecs, fts, word_mask, ft_mask, is_trn)
      sim = tst(wvecs, fts, word_mask, ft_mask, is_trn)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_word_sim,
      }
    else:
      sim = tst(wvecs, fts, word_mask, ft_mask, is_trn)
      return {
        self.OutKey.SIM: sim,
      }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      pos_sim = self._outputs[self.OutKey.P_SIM]
      neg_caption_sim = self._outputs[self.OutKey.NC_SIM]
      neg_ft_sim = self._outputs[self.OutKey.NF_SIM]
      self.op2monitor['pos_sim'] = tf.reduce_mean(pos_sim)
      self.op2monitor['neg_caption_sim'] = tf.reduce_mean(neg_caption_sim)
      self.op2monitor['neg_ft_sim'] = tf.reduce_mean(neg_ft_sim)

      contrast_caption_loss = neg_caption_sim + self._config.margin - pos_sim
      contrast_caption_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_caption_loss))
      # self.op2monitor['contrast_caption_loss'] = tf.reduce_sum(contrast_caption_loss)

      contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
      contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))
      # self.op2monitor['contrast_ft_loss'] = tf.reduce_sum(contrast_ft_loss)

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


PathCfg = trntst_util.AttPathCfg
TrnTst = trntst_util.AttTrnTst

TrnReader = trntst_util.TrnAttReader
ValReader = trntst_util.ValAttReader
TstReader = trntst_util.TstAttReader

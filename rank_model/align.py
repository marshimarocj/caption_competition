import os
import sys
import json
import cPickle
import random
import enum
sys.path.append('../')

import tensorflow as tf
import numpy as np
import mosek

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

  cfg.max_words_in_caption = kwargs['max_words_in_caption']

  we_cfg = cfg.subcfgs[WE]
  we_cfg.lr_mult = kwargs['lr_mult']

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'align.Model'

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
      self.InKey.CAPTIONID: captionids,
      self.InKey.FT_MASK: ft_masks,
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

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[WE]
    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.CAPTION: in_ops[self.InKey.CAPTIONID],
      }, mode)
    wvecs = out_ops[encoder.OutKey.EMBED] # (None, max_words_in_caption, dim_embed)
    fts = in_ops[self.InKey.FT]
    is_trn = in_ops[self.InKey.IS_TRN]
    word_masks = in_ops[self.InKey.CAPTION_MASK]
    ft_masks = in_ops[self.InKey.FT_MASK]

    with tf.variable_scope(self.name_scope):
      fts = tf.reshape(fts, (-1, dim_ft))
      fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[0], self.ft_pca_Bs[0])
      fts = tf.nn.relu(fts)
      fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[1], self.ft_pca_Bs[1])
      fts = tf.reshape(fts, (-1, num_ft, dim_embed))
      fts = tf.nn.l2_normalize(fts, -1)

      wvecs = tf.reshape(wvecs, (-1, dim_word))
      wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[0], self.caption_pca_Bs[0])
      wvecs = tf.nn.relu(wvecs)
      wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[1], self.caption_pca_Bs[1])
      wvecs = tf.reshape(wvecs, (-1, num_word, dim_embed))
      wvecs = tf.nn.l2_normalize(wvecs, -1)

      batch_size = tf.shape(wvecs)[0]
      num_pos = batch_size - self._config.num_neg
      num_neg = self._config.num_neg
      num_word = self._config.max_words_in_caption
      dim_word = self._config.subcfgs[WE].dim_embed
      dim_ft = self._config.dim_ft
      num_ft = self._config.num_track
      dim_embed = self._config.dim_joint_embed

    def trn(wvecs, fts, word_masks, ft_masks, is_trn):
      with tf.variable_scope(self.name_scope):
        pos_fts = fts[:num_pos]
        neg_fts = fts[num_pos:]
        pos_wvecs = wvecs[:num_pos]
        neg_wvecs = wvecs[num_pos:]
        pos_word_masks = word_masks[:num_pos]
        neg_word_masks = word_masks[num_pos:]
        pos_ft_masks = ft_masks[:num_pos]
        neg_ft_masks = ft_masks[num_pos:]

        pos_sim = tf.reduce_sum(tf.expand_dims(pos_fts, 2) * tf.expand_dims(pos_wvecs, 1), -1) # (num_pos, num_ft, num_word)
        pos_sim = tf.reduce_sum(pos_sim, tf.expand_dims(pos_word_masks, 1) * tf.expand_dims(pos_ft_masks, 2), [1, 2])
        pos_sim /= tf.reduce_sum(pos_word_masks, 1) * tf.reduce_sum(pos_ft_masks, 1)

        expand_pos_fts = tf.reshape(pos_fts, (num_pos, num_ft, 1, 1, dim_embed))
        expand_neg_wvecs = tf.reshape(neg_wvecs, (1, 1, num_neg, num_word, dim_embed))
        neg_word_sim = tf.reduce_sum(expand_pos_fts * expand_neg_wvecs, -1) # (num_pos, num_ft, num_neg, num_word)
        neg_word_sim = tf.transpose(neg_word_sim, (0, 2, 1, 3)) # (num_pos, num_neg, num_ft, num_word)
        neg_word_sim = tf.reduce_sum(
          neg_word_sim * tf.reshape(pos_ft_masks, (num_pos, 1, num_ft, 1)) * tf.reshape(neg_word_masks, (1, num_neg, 1, num_word)), [2, 3])
        neg_word_sim /= tf.expand_dims(tf.reduce_sum(pos_ft_masks, 1), 1)
        neg_word_sim /= tf.expand_dims(tf.reduce_sum(neg_word_masks, 1), 0)
        neg_word_sim = tf.reduce_logsumexp(100.*neg_word_sim, 1) / 100.

        expand_neg_fts = tf.reshape(neg_fts, (num_neg, num_ft, 1, 1, dim_embed))
        expand_pos_wvecs = tf.reshape(pos_wvecs, (1, 1, num_pos, num_word, dim_embed))
        neg_ft_sim = tf.reduce_sum(expand_neg_fts * expand_pos_wvecs, -1) # (num_neg, num_ft, num_pos, num_word)
        neg_ft_sim = tf.transpose(neg_ft_sim, (0, 2, 1, 3)) # (num_neg, num_pos, num_ft, num_word)
        neg_ft_sim = tf.reduce_sum(
          neg_ft_sim * tf.reshape(neg_ft_masks, (num_neg, 1, num_ft, 1)) * tf.reshape(pos_ft_masks, (1, num_pos, 1, num_word)), [2, 3])
        neg_ft_sim /= tf.expand_dims(tf.reduce_sum(neg_ft_masks, 1), 1)
        neg_ft_sim /= tf.expand_dims(tf.reduce_sum(pos_word_masks, 1), 0)
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 0) / 100.

        return pos_sim, neg_word_sim, neg_ft_sim

    def tst(wvecs, fts, word_masks, ft_masks, is_trn):
      with tf.variable_scope(self.name_scope):
        fts = tf.reshape(fts, (-1, num_ft, 1, 1, dim_embed))
        wvecs = tf.reshape(wvecs, (1, 1, -1, num_word, dim_embed))
        sim = tf.reduce_sum(fts * wvecs, -1) # (None, num_ft, None, num_word)
        sim = tf.transpose(sim, [0, 2, 1, 3]) # (None, None, num_ft, num_word)
        sim = tf.reduce_sum(
          sim * tf.reshape(ft_masks, (-1, 1, num_ft, 1)) * tf.reshape(word_masks, (1, -1, 1, num_word)), [2, 3])
        sim /= tf.expand_dims(tf.reduce_sum(ft_masks, 1), 1)
        sim /= tf.expand_dims(tf.reduce_sum(word_masks, 1), 0)

        return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      pos_sim, neg_word_sim, neg_ft_sim = trn(wvecs, fts, word_masks, ft_masks, is_trn)
      sim = tst(wvecs, fts, word_masks, ft_masks, is_trn)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_word_sim,
      }
    else:
      sim = tst(wvecs, fts, word_masks, ft_masks, is_trn)
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

      contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
      contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))

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


class PGModel(framework.model.module.AbstractPGModel):
  name_scope = 'align.Model'

  class InKey(enum.Enum):
    FT = 'ft'
    # FT_MASK = 'ft_mask'
    CAPTIONID = 'captionid'
    # CAPTION_MASK = 'caption_mask'
    ALIGN = 'align'
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
      # ft_masks = tf.placeholder(
      #   tf.float32, shape=(None, self._config.num_track), name='ft_masks')
      captionids = tf.placeholder(
        tf.int32, shape=(None, self._config.max_words_in_caption), name='captionids')
      # caption_masks = tf.placeholder(
      #   tf.int32, shape=(None, self._config.max_words_in_caption), name='caption_masks')
      is_trn = tf.placeholder(tf.bool, shape=(), name='is_trn')
      align = tf.placeholder(
        tf.float32, shape=(None, self._config.num_track, self._config.max_words_in_caption), name='align') # (num_pos + 2*num_pos*num_neg)

    return {
      self.InKey.FT: fts,
      self.InKey.CAPTIONID: captionids,
      # self.InKey.CAPTION_MASK: caption_masks,
      self.InKey.IS_TRN: is_trn,
      self.InKey.ALIGN: align,
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

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[WE]
    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.CAPTION: in_ops[self.InKey.CAPTIONID],
      }, mode)
    wvecs = out_ops[encoder.OutKey.EMBED] # (None, max_words_in_caption, dim_embed)
    fts = in_ops[self.InKey.FT]
    is_trn = in_ops[self.InKey.IS_TRN]

    with tf.variable_scope(self.name_scope):
      fts = tf.reshape(fts, (-1, dim_ft))
      fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[0], self.ft_pca_Bs[0])
      fts = tf.nn.relu(fts)
      fts = tf.nn.xw_plus_b(fts, self.ft_pca_Ws[1], self.ft_pca_Bs[1])
      fts = tf.reshape(fts, (-1, num_ft, dim_embed))
      fts = tf.nn.l2_normalize(fts, -1)

      wvecs = tf.reshape(wvecs, (-1, dim_word))
      wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[0], self.caption_pca_Bs[0])
      wvecs = tf.nn.relu(wvecs)
      wvecs = tf.nn.xw_plus_b(wvecs, self.caption_pca_Ws[1], self.caption_pca_Bs[1])
      wvecs = tf.reshape(wvecs, (-1, num_word, dim_embed))
      wvecs = tf.nn.l2_normalize(wvecs, -1)

      batch_size = tf.shape(wvecs)[0]
      num_pos = batch_size - self._config.num_neg
      num_neg = self._config.num_neg
      num_word = self._config.max_words_in_caption
      dim_word = self._config.subcfgs[WE].dim_embed
      dim_ft = self._config.dim_ft
      num_ft = self._config.num_track
      dim_embed = self._config.dim_joint_embed

    def rollout(wvecs, fts, is_trn):
      with tf.variable_scope(self.name_scope):
        pos_fts = fts[:num_pos]
        neg_fts = fts[num_pos:]
        pos_wvecs = wvecs[:num_pos]
        neg_wvecs = wvecs[num_pos:]

        pos_sim = tf.reduce_sum(tf.expand_dims(pos_fts, 2) * tf.expand_dims(pos_wvecs, 1), -1) # (num_pos, num_ft, num_word)

        expand_pos_fts = tf.reshape(pos_fts, (num_pos, num_ft, 1, 1, dim_embed))
        expand_neg_wvecs = tf.reshape(neg_wvecs, (1, 1, num_neg, num_word, dim_embed))
        neg_word_sim = tf.reduce_sum(expand_pos_fts * expand_neg_wvecs, -1) # (num_pos, num_ft, num_neg, num_word)
        neg_word_sim = tf.transpose(neg_word_sim, (0, 2, 1, 3)) # (num_pos, num_neg, num_ft, num_word)
        neg_word_sim = tf.reshape(neg_word_sim, (-1, num_ft, num_word))

        expand_neg_fts = tf.reshape(neg_fts, (num_neg, num_ft, 1, 1, dim_embed))
        expand_pos_wvecs = tf.reshape(pos_wvecs, (1, 1, num_pos, num_word, dim_embed))
        neg_ft_sim = tf.reduce_sum(expand_neg_fts * expand_pos_wvecs, -1) # (num_neg, num_ft, num_pos, num_word)
        neg_ft_sim = tf.transpose(neg_ft_sim, (0, 2, 1, 3)) # (num_neg, num_pos, num_ft, num_word)
        neg_ft_sim = tf.reshape(neg_ft_sim, (-1, num_ft, num_word))

        sim = tf.concat([pos_sim, neg_word_sim, neg_ft_sim], 0)

        return sim

    def trn(wvecs, fts, align, is_trn):
      with tf.variable_scope(self.name_scope):
        pos_fts = fts[:num_pos]
        neg_fts = fts[num_pos:]
        pos_wvecs = wvecs[:num_pos]
        neg_wvecs = wvecs[num_pos:]

        pos_sim = tf.reduce_sum(tf.expand_dims(pos_fts, 2) * tf.expand_dims(pos_wvecs, 1), -1) # (num_pos, num_ft, num_word)
        pos_sim = tf.reduce_sum(pos_sim * align[:num_pos], [1, 2])

        expand_pos_fts = tf.reshape(pos_fts, (num_pos, num_ft, 1, 1, dim_embed))
        expand_neg_wvecs = tf.reshape(neg_wvecs, (1, 1, num_neg, num_word, dim_embed))
        neg_word_sim = tf.reduce_sum(expand_pos_fts * expand_neg_wvecs, -1) # (num_pos, num_ft, num_neg, num_word)
        neg_word_sim = tf.transpose(neg_word_sim, (0, 2, 1, 3)) # (num_pos, num_neg, num_ft, num_word)
        neg_word_align = align[num_pos:num_pos+num_pos*num_neg]
        neg_word_align = tf.reshape(neg_word_align, (num_pos, num_neg, num_ft, num_word))
        neg_word_sim = tf.reduce_sum(neg_word_sim * neg_word_align, [2, 3])
        neg_word_sim = tf.reduce_logsumexp(100.*neg_word_sim, 1) / 100.

        expand_neg_fts = tf.reshape(neg_fts, (num_neg, num_ft, 1, 1, dim_embed))
        expand_pos_wvecs = tf.reshape(pos_wvecs, (1, 1, num_pos, num_word, dim_embed))
        neg_ft_sim = tf.reduce_sum(expand_neg_fts * expand_pos_wvecs, -1) # (num_neg, num_ft, num_pos, num_word)
        neg_ft_sim = tf.transpose(neg_ft_sim, (0, 2, 1, 3)) # (num_neg, num_pos, num_ft, num_word)
        neg_ft_align = align[num_pos+num_pos*num_neg:]
        neg_ft_align = tf.reshape(neg_ft_align, (num_neg, num_pos, num_ft, num_word))
        neg_ft_sim = tf.reduce_sum(neg_ft_sim * neg_ft_align, [2, 3])
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 0) / 100.

        return pos_sim, neg_word_sim, neg_ft_sim

    def tst(wvecs, fts, is_trn):
      with tf.variable_scope(self.name_scope):
        fts = tf.reshape(fts, (-1, num_ft, 1, 1, dim_embed))
        wvecs = tf.reshape(wvecs, (1, 1, -1, num_word, dim_embed))
        sim = tf.reduce_sum(fts * wvecs, -1) # (None, num_ft, None, num_word)
        sim = tf.transpose(sim, [0, 2, 1, 3]) # (None, None, num_ft, num_word)

        return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      align = in_ops[self.InKey.ALIGN]
      pos_sim, neg_word_sim, neg_ft_sim = trn(wvecs, fts, align, is_trn)
      sim = tst(wvecs, fts, is_trn)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_word_sim,
      }
    elif mode == framework.model.module.Mode.ROLLOUT:
      sim = rollout(wvecs, fts, is_trn)
      return {
        self.OutKey.SIM: sim,
      }
    else:
      sim = tst(wvecs, fts, is_trn)
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

      contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
      contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))

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

  def op_in_rollout(self, **kwargs):
    op_dict = {
      self.OutKey.SIM: self._outputs[self.OutKey.SIM],
    }


PathCfg = trntst_util.AttPathCfg
TrnTst = trntst_util.AttTrnTst


class PGTrnTst(framework.model.trntst.TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      # self.model.inputs[self.model.InKey.FT_MASK]: data['ft_masks'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      # self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
      self.model.inputs[self.model.InKey.ALIGN]: data['aligns'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }

  def _construct_feed_dict_in_rollout(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      # self.model.inputs[self.model.InKey.FT_MASK]: data['ft_masks'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      # self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
      self.model.inputs[self.model.InKey.IS_TRN]: False,
    }

  def feed_data_and_rollout(self, data, sess):
    op_dict = self.model.op_in_rollout()

    feed_dict = self._construct_feed_dict_in_rollout(data)
    sim = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)

    batch_size = data['fts'].shape[0]
    num_neg = self.model_cfg.neg
    num_pos = batch_size - num_neg

    ft_masks = data['ft_masks']
    pos_ft_masks = ft_masks[:num_pos]
    neg_ft_masks = ft_masks[num_pos:]

    caption_masks = data['caption_masks']
    pos_caption_masks = caption_masks[:num_pos]
    neg_caption_masks = caption_masks[num_pos:]

    aligns = []
    with mosek.Env() as env:
      pos_sim = sim[:num_pos]
      for i in range(num_pos):
        align = max_match(pos_sim[i], pos_ft_masks[i], pos_caption_masks[i], env)
        aligns.append(align)

      neg_word_sim = sim[num_pos:num_pos+num_pos*num_neg]
      neg_word_sim = neg_word_sim.reshape((num_pos, num_neg))
      for i in range(num_pos):
        for j in range(num_neg):
          align = max_match(neg_word_sim[i, j], pos_ft_masks[i], neg_caption_masks[j], env)
          aligns.append(align)

      neg_ft_sim = sim[num_pos+num_pos*num_neg:]
      neg_ft_sim = neg_ft_sim.reshape((num_neg, num_pos))
      for i in range(num_neg):
        for j in range(num_pos):
          align = max_match(neg_ft_sim[i, j], neg_ft_masks[i], pos_caption_masks[j], env)
          aligns.append(align)
    aligns = np.array(aligns, dtype=np.float32)
    data['aligns'] = aligns

    return data

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    mir = 0.
    num = 0
    for data in tst_reader.yield_val_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        # self.model.inputs[self.model.InKey.FT_MASK]: data['ft_masks'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        # self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      ft_masks = data['ft_masks']
      caption_masks = data['caption_masks']
      sims = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      scores = []
      with mosek.Env() as env:
        for i, sim in enumerate(sims[0]):
          align = max_match(sim, ft_masks[0], caption_masks[i], env)
          score = np.sum(sim * align)
          scores.append(score)
      idxs = np.argsort(-scores)
      rank = np.where(idxs == data['gt'])[0][0]
      rank += 1
      mir += 1. / rank
      num += 1
    mir /= num
    metrics['mir'] = mir

  def predict_in_tst(self, sess, tst_reader, predict_file):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    all_scores = []
    for data in tst_reader.yield_tst_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        # self.model.inputs[self.model.InKey.FT_MASK]: data['ft_masks'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        # self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      ft_masks = data['ft_masks']
      caption_masks = data['caption_masks']
      sim = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      scores = []
      with mosek.Env() as env:
        for i, sim in enumerate(sims[0]):
          align = max_match(sim, ft_masks[0], caption_masks[i], env)
          score = np.sum(sim * align)
          scores.append(score)
      all_scores.append(scores)
    np.save(predict_file, all_scores)


def max_match(sim, row_mask, col_mask, env):
  rows = np.sum(row_mask)
  cols = np.sum(col_mask)
  C = sim[:rows, :cols]
  C = C.reshape((-1,))
  align = np.zeros(sim.shape, dtype=np.float32)

  with env.Task(0, 0) as task:
    task.appendcons(rows)
    task.appendvars(rows * cols)
    for i, c in C:
      task.putcj(i, c)
    for i in range(rows*cols):
      task.putvarbound(i, mosek.boundkey.ra, 0., 1.)
    for i in range(rows):
      task.putarow(i, range(i*cols, (i+1)*cols), [1.]*cols)
    for i in range(rows):
      task.putconbound(i, mosek.boundkey.fx, 1., 1.)

    task.putobjsense(mosek.objsense.maximize)
    task.optimize()

    solsta = task.getsolsta(mosek.soltype.bas)

    if (solsta == mosek.solsta.optimal or
          solsta == mosek.solsta.near_optimal):
      xx = [0.] * (rows * cols)
      task.getxx(mosek.soltype.bas, # Request the basic solution.
                 xx)
      xx = np.array(xx)
      xx = xx.reshape((rows, cols))
      align[:rows, :cols] += xx

  with env.Task(0, 0) as task:
    task.appendcons(cols)
    task.appendvars(rows * cols)
    for i, c in C:
      task.putcj(i, c)
    for i in range(rows*cols):
      task.putvarbound(i, mosek.boundkey.ra, 0., 1.)
    for i in range(cols):
      task.putarow(i, range(i, rows*cols, cols), [1.]*rows)
    for i in range(cols):
      task.putconbound(i, mosek.boundkey.fx, 1., 1.)

    task.putobjsense(mosek.objsense.maximize)
    task.optimize()

    solsta = task.getsolsta(mosek.soltype.bas)

    if (solsta == mosek.solsta.optimal or
          solsta == mosek.solsta.near_optimal):
      xx = [0.] * (rows * cols)
      task.getxx(mosek.soltype.bas, # Request the basic solution.
                 xx)
      xx = np.array(xx)
      xx = xx.reshape((rows, cols))
      align[:rows, :cols] += xx
  align /= rows + cols

  return align

TrnReader = trntst_util.AttTrnReader
ValReader = trntst_util.AttValReader
TstReader = trntst_util.AttTstReader

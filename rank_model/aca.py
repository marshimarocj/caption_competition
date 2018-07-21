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
    # self.tanh_scale = 1.


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
  # cfg.tanh_scale = kwargs['tanh_scale']

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
      self.word_att_W = tf.contrib.framework.model_variable('word_att_W',
        shape=(self._config.subcfgs[WE].dim_embed, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.word_att_B = tf.contrib.framework.model_variable('word_att_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.word_att_W)
      self._weights.append(self.word_att_B)

      self.ft_att_W = tf.contrib.framework.model_variable('ft_att_W',
        shape=(self._config.dim_ft, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.ft_att_B = tf.contrib.framework.model_variable('ft_att_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.ft_att_W)
      self._weights.append(self.ft_att_B)

      self.compare_W = tf.contrib.framework.model_variable('compare_W',
        shape=(self._config.subcfgs[WE].dim_embed + self._config.dim_ft, self._config.dim_joint_embed), 
        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
      self.compare_B = tf.contrib.framework.model_variable('compare_B',
        shape=(self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.compare_W)
      self._weights.append(self.compare_B)

      # dim_inputs = [self._config.dim_joint_embed*2, self._config.dim_joint_embed]
      # dim_outputs =[self._config.dim_joint_embed, 1]
      # layer = 0
      # for dim_input, dim_output in zip(dim_inputs, dim_outputs):
      #   W = tf.contrib.framework.model_variable('aggregate_W_%d'%layer,
      #     shape=(dim_input, dim_output), dtype=tf.float32,
      #     initializer=tf.contrib.layers.xavier_initializer())
      #   B = tf.contrib.framework.model_variable('aggregate_B_%d'%layer,
      #     shape=(dim_output,), dtype=tf.float32,
      #     initializer=tf.random_uniform_initializer(-0.1, 0.1))
      #   self.aggregate_Ws.append(W)
      #   self.aggregate_Bs.append(B)
      #   self._weights.append(W)
      #   self._weights.append(B)
      #   layer += 1

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[WE]
    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.CAPTION: in_ops[self.InKey.CAPTIONID],
      }, mode)
    wvecs = out_ops[encoder.OutKey.EMBED] # (None, max_words_in_caption, dim_embed)
    fts = in_ops[self.InKey.FT]
    mask = in_ops[self.InKey.CAPTION_MASK]
    is_trn = in_ops[self.InKey.IS_TRN]

    def trn(wvecs, fts, mask, is_trn):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(wvecs)[0]
        num_pos = batch_size - self._config.num_neg
        num_neg = self._config.num_neg
        num_word = self._config.max_words_in_caption
        dim_word = self._config.subcfgs[WE].dim_embed
        dim_ft = self._config.dim_ft
        dim_embed = self._config.dim_joint_embed

        pos_fts = fts[:num_pos]
        neg_fts = fts[num_pos:]
        pos_wvecs = wvecs[:num_pos]
        neg_wvecs = wvecs[num_pos:]
        mask = tf.to_float(mask)
        pos_mask = mask[:num_pos]
        neg_mask = mask[num_pos:]

        alpha = tf.nn.xw_plus_b(tf.reshape(wvecs, (-1, dim_word)), self.word_att_W, self.word_att_B)
        alpha = tf.nn.relu(alpha) # (None, num_word, dim_embed)
        alpha = tf.reshape(alpha, (-1, num_word, dim_embed))
        beta = tf.nn.xw_plus_b(fts, self.ft_att_W, self.ft_att_B)
        beta = tf.nn.relu(beta)
        beta = tf.expand_dims(beta, 1) # (None, 1, dim_embed)
        pos_alpha = alpha[:num_pos]
        neg_alpha = alpha[num_pos:]
        pos_beta = beta[:num_pos]
        neg_beta = beta[num_pos:]

        def calc_pos_sim(pos_fts, pos_wvecs, pos_alpha, pos_beta, pos_mask):
        # def calc_pos_sim():
          # attend
          att = tf.matmul(pos_alpha, pos_beta, transpose_b=True) # (num_pos, num_word, 1)
          att = att[:, :, 0] # (num_pos, num_word)
          att = tf.nn.softmax(att, 1)
          att *= pos_mask
          att /= tf.reduce_sum(att, 1, True)
          wvec_bar = tf.reduce_sum(pos_wvecs * tf.expand_dims(att, 2), 1) # (num_pos, dim_word)

          # compare
          expanded_fts = tf.tile(tf.expand_dims(pos_fts, 1), [1, num_word, 1]) # (num_pos, num_word, dim_ft)
          wvec_ft = tf.concat([
            tf.reshape(pos_wvecs, (-1, dim_word)), # (num_pos*num_word, dim_word)
            tf.reshape(expanded_fts, (-1, dim_ft)) # (num_pos*num_word, dim_ft)
            ], 1)
          wvec_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
          # wvec_compare = tf.layers.dropout(wvec_compare, training=is_trn)
          wvec_compare = tf.nn.relu(wvec_compare) # (num_pos*num_word, dim_embed)
          wvec_compare = tf.reshape(wvec_compare, (-1, num_word, dim_embed))

          wvec_ft = tf.concat([wvec_bar, pos_fts], 1)
          ft_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
          # ft_compare = tf.layers.dropout(ft_compare, training=is_trn)
          ft_compare = tf.nn.relu(ft_compare) # (num_pos, dim_embed)

          # aggregate
          pos_mask = tf.expand_dims(pos_mask, 2)
          wvec_aggregate = tf.reduce_sum(wvec_compare * pos_mask, 1) / tf.reduce_sum(pos_mask, 1)
          wvec_aggregate = tf.nn.l2_normalize(wvec_aggregate, -1)
          ft_compare = tf.nn.l2_normalize(ft_compare, -1)
          # pos_sim = tf.concat([wvec_aggregate, ft_compare], 1)
          # pos_sim = tf.nn.xw_plus_b(pos_sim, self.aggregate_Ws[0], self.aggregate_Bs[0])
          # pos_sim = tf.nn.relu(pos_sim)
          # pos_sim = tf.nn.xw_plus_b(pos_sim, self.aggregate_Ws[1], self.aggregate_Bs[1])
          # # pos_sim = tf.tanh(pos_sim / self._config.tanh_scale) # (num_pos, 1)
          # pos_sim = tf.reshape(pos_sim, (num_pos,))
          pos_sim = tf.reduce_sum(wvec_aggregate * ft_compare, -1)

          return pos_sim

        def calc_neg_word_sim(pos_fts, neg_wvecs, neg_alpha, pos_beta, neg_mask):
          # attend
          neg_alpha = tf.reshape(neg_alpha, (-1, dim_embed)) # (num_neg*num_word, dim_embed)
          pos_beta = tf.reshape(pos_beta, (-1, dim_embed)) # (num_pos, dim_embed)
          att = tf.matmul(neg_alpha, pos_beta, transpose_b=True)
          att = tf.reshape(att, (num_neg, num_word, num_pos)) # (num_neg, num_word, num_pos)
          att = tf.nn.softmax(att, 1)
          att *= tf.expand_dims(neg_mask, 2)
          att /= tf.reduce_sum(att, 1, True)
          wvecs_bar = tf.reduce_sum(
            tf.expand_dims(neg_wvecs, 2) * tf.expand_dims(att, 3), 1) # (num_neg, num_pos, dim_word)

          # compare
          expanded_fts = tf.tile(
            tf.reshape(pos_fts, (1, -1, 1, dim_ft)), 
            [num_neg, 1, num_word, 1]) # (num_neg, num_pos, num_word, dim_ft)
          expanded_wvecs = tf.tile(
            tf.expand_dims(neg_wvecs, 1), [1, num_pos, 1, 1]) # (num_neg, num_pos, num_word, dim_word)
          wvec_ft = tf.reshape(tf.concat([expanded_wvecs, expanded_fts], 3), (-1, dim_word + dim_ft))
          wvec_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
          # wvec_compare = tf.layers.dropout(wvec_compare, training=is_trn)
          wvec_compare = tf.nn.relu(wvec_compare)
          wvec_compare = tf.reshape(wvec_compare, (num_neg, num_pos, num_word, dim_embed))

          expanded_fts = tf.tile(tf.expand_dims(pos_fts, 0), [num_neg, 1, 1]) # (num_neg, num_pos, dim_ft)
          wvec_ft = tf.reshape(tf.concat([wvecs_bar, expanded_fts], 2), (-1, dim_word + dim_ft))
          ft_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
          # ft_compare = tf.layers.dropout(ft_compare, training=is_trn)
          ft_compare = tf.nn.relu(ft_compare)
          ft_compare = tf.reshape(ft_compare, (num_neg, num_pos, dim_embed))

          # aggregate
          neg_mask = tf.reshape(neg_mask, (num_neg, 1, num_word, 1))
          wvec_aggregate = tf.reduce_sum(wvec_compare * neg_mask, 2) / tf.reduce_sum(neg_mask, 2)
          wvec_aggregate = tf.nn.l2_normalize(wvec_aggregate, -1)
          ft_compare = tf.nn.l2_normalize(ft_compare, -1)
          # neg_sim = tf.reshape(tf.concat([wvec_aggregate, ft_compare], 2), (-1, dim_embed*2))
          # neg_sim = tf.nn.xw_plus_b(neg_sim, self.aggregate_Ws[0], self.aggregate_Bs[0])
          # neg_sim = tf.nn.relu(neg_sim)
          # neg_sim = tf.nn.xw_plus_b(neg_sim, self.aggregate_Ws[1], self.aggregate_Bs[1])
          # # neg_sim = tf.tanh(neg_sim / self._config.tanh_scale) # (num_neg*num_pos, 1)
          # neg_sim = tf.reshape(neg_sim, (num_neg, num_pos))
          neg_sim = tf.reduce_sum(wvec_aggregate * ft_compare, -1)
          neg_sim = tf.reshape(neg_sim, (num_neg, num_pos))

          return neg_sim

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
            tf.expand_dims(pos_wvecs, 0) * tf.expand_dims(att, 3), 2) # (num_neg, num_pos, dim_word)

          # compare
          expanded_fts = tf.tile(
            tf.reshape(neg_fts, (-1, 1, 1, dim_ft)),
            [1, num_pos, num_word, 1]) # (num_neg, num_pos, num_word, dim_ft)
          expanded_wvecs = tf.tile(
            tf.expand_dims(pos_wvecs, 0), [num_neg, 1, 1, 1]) # (num_neg, num_pos, num_word, dim_word)
          wvec_ft = tf.reshape(tf.concat([expanded_wvecs, expanded_fts], 3), (-1, dim_word + dim_ft))
          wvec_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
          # wvec_compare = tf.layers.dropout(wvec_compare, training=is_trn)
          wvec_compare = tf.nn.relu(wvec_compare)
          wvec_compare = tf.reshape(wvec_compare, (num_neg, num_pos, num_word, dim_embed))

          expanded_fts = tf.tile(
            tf.reshape(neg_fts, (-1, 1, dim_ft)),
            [1, num_pos, 1]) # (num_neg, num_pos, dim_ft)
          wvec_ft = tf.reshape(tf.concat([wvecs_bar, expanded_fts], 2), (-1, dim_word + dim_ft))
          ft_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
          # ft_compare = tf.layers.dropout(ft_compare, training=is_trn)
          ft_compare = tf.nn.relu(ft_compare)
          ft_compare = tf.reshape(ft_compare, (num_neg, num_pos, dim_embed))

          # aggregate
          pos_mask = tf.reshape(pos_mask, (1, num_pos, num_word, 1))
          wvec_aggregate = tf.reduce_sum(wvec_compare * pos_mask, 2) / tf.reduce_sum(pos_mask, 2)
          wvec_aggregate = tf.nn.l2_normalize(wvec_aggregate, -1)
          ft_compare = tf.nn.l2_normalize(ft_compare, -1)
          # neg_sim = tf.reshape(tf.concat([wvec_aggregate, ft_compare], 2), (-1, dim_embed*2))
          # neg_sim = tf.nn.xw_plus_b(neg_sim, self.aggregate_Ws[0], self.aggregate_Bs[0])
          # neg_sim = tf.nn.relu(neg_sim)
          # neg_sim = tf.nn.xw_plus_b(neg_sim, self.aggregate_Ws[1], self.aggregate_Bs[1])
          # # neg_sim = tf.tanh(neg_sim / self._config.tanh_scale) # (num_neg*num_pos, 1)
          # neg_sim = tf.reshape(neg_sim, (num_neg, num_pos))
          neg_sim = tf.reduce_sum(wvec_aggregate * ft_compare, -1)
          neg_sim = tf.reshape(neg_sim, (num_neg, num_pos))

          return neg_sim

        pos_sim = calc_pos_sim(pos_fts, pos_wvecs, pos_alpha, pos_beta, pos_mask)
        neg_word_sim = calc_neg_word_sim(pos_fts, neg_wvecs, neg_alpha, pos_beta, neg_mask)
        neg_ft_sim = calc_neg_ft_sim(neg_fts, pos_wvecs, pos_alpha, neg_beta, pos_mask)
        neg_word_sim = tf.reduce_logsumexp(100.*neg_word_sim, 0) / 100. # (num_pos,)
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 0) / 100.

        return pos_sim, neg_word_sim, neg_ft_sim

    def tst(wvecs, fts, mask, is_trn):
      with tf.variable_scope(self.name_scope):
        num_ft = tf.shape(fts)[0]
        num_caption = tf.shape(wvecs)[0]
        num_word = self._config.max_words_in_caption
        dim_word = self._config.subcfgs[WE].dim_embed
        dim_ft = self._config.dim_ft
        dim_embed = self._config.dim_joint_embed
        mask = tf.to_float(mask)

        # attend
        alpha = tf.nn.xw_plus_b(tf.reshape(wvecs, (-1, dim_word)), self.word_att_W, self.word_att_B)
        alpha = tf.nn.tanh(alpha) # (num_caption*num_word, dim_embed)
        beta = tf.nn.xw_plus_b(fts, self.ft_att_W, self.ft_att_B)
        beta = tf.nn.tanh(beta) # (num_ft, dim_embed)

        att = tf.matmul(beta, alpha, transpose_b=True) # (num_ft, num_caption*num_word)
        att = tf.reshape(att, (num_ft, num_caption, num_word))
        att = tf.nn.softmax(att, 2)
        att *= tf.expand_dims(mask, 0)
        att /= tf.reduce_sum(att, 2, True)
        wvecs_bar = tf.reduce_sum(
          tf.expand_dims(wvecs, 0) * tf.expand_dims(att, 3), 2) # (num_ft, num_caption, dim_word)

        # compare
        expanded_fts = tf.tile(
          tf.reshape(fts, (-1, 1, 1, dim_ft)),
          [1, num_caption, num_word, 1]) # (num_ft, num_caption, num_word, dim_ft)
        expanded_wvecs = tf.tile(
          tf.expand_dims(wvecs, 0), [num_ft, 1, 1, 1]) # (num_ft, num_caption, num_word, dim_word)
        wvec_ft = tf.reshape(tf.concat([expanded_wvecs, expanded_fts], 3), (-1, dim_word + dim_ft))
        wvec_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
        wvec_compare = tf.layers.dropout(wvec_compare, training=is_trn)
        wvec_compare = tf.nn.relu(wvec_compare)
        wvec_compare = tf.reshape(wvec_compare, (num_ft, num_caption, num_word, dim_embed))

        expanded_fts = tf.tile(
          tf.reshape(fts, (-1, 1, dim_ft)),
          [1, num_caption, 1]) # (num_ft, num_caption, dim_ft)
        wvec_ft = tf.reshape(tf.concat([wvecs_bar, expanded_fts], 2), (-1, dim_word + dim_ft))
        ft_compare = tf.nn.xw_plus_b(wvec_ft, self.compare_W, self.compare_B)
        ft_compare = tf.layers.dropout(ft_compare, training=is_trn)
        ft_compare = tf.nn.relu(ft_compare)
        ft_compare = tf.reshape(ft_compare, (num_ft, num_caption, dim_embed))

        # aggregate
        mask = tf.reshape(mask, (1, num_caption, num_word, 1))
        wvec_aggregate = tf.reduce_sum(wvec_compare * mask, 2) / tf.reduce_sum(mask, 2)
        wvec_aggregate = tf.nn.l2_normalize(wvec_aggregate, -1)
        ft_compare = tf.nn.l2_normalize(ft_compare, -1)
        # sim = tf.reshape(tf.concat([wvec_aggregate, ft_compare], 2), (-1, dim_embed*2))
        # sim = tf.nn.xw_plus_b(sim, self.aggregate_Ws[0], self.aggregate_Bs[0])
        # sim = tf.nn.relu(sim)
        # sim = tf.nn.xw_plus_b(sim, self.aggregate_Ws[1], self.aggregate_Bs[1])
        # # sim = tf.tanh(sim / self._config.tanh_scale)
        # sim = tf.reshape(sim, (num_ft, num_caption))
        sim = tf.reduce_sum(wvec_aggregate * ft_compare, -1)
        sim = tf.reshape(sim, (num_ft, num_caption))

        return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      pos_sim, neg_word_sim, neg_ft_sim = trn(wvecs, fts, mask, is_trn)
      sim = tst(wvecs, fts, mask, is_trn)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_word_sim,
      }
    else:
      sim = tst(wvecs, fts, mask, is_trn)
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


PathCfg = trntst_util.PathCfg
TrnTst = trntst_util.TrnTst

TrnReader = trntst_util.TrnReader
ValReader = trntst_util.ValReader
TstReader = trntst_util.TstReader

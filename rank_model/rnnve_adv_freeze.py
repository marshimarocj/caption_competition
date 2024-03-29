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
import trntst_util
import adv.transform


ADV = 'adversarial'


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[ADV] = adv.transform.Config()

    self.max_words_in_caption = 30

    self.dim_ft = 1024
    self.dim_joint_embeds = [170, 171, 171]
    self.dim_caption = 500
    self.margin = 0.1
    self.alpha = 0.5
    self.num_neg = 1
    self.l2norm = True
    self.beta = 100.
    self.pool = 'max'

  def _assert(self):
    assert len(self.dim_joint_embeds) == len(self.subcfgs[ADV].dim_inputs)
    for dim_joint_embed, dim_input in zip(self.dim_joint_embeds, self.subcfgs[ADV].dim_inputs):
      assert dim_joint_embed == dim_input


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
  cfg.beta = kwargs['beta']
  cfg.num_neg = kwargs['num_neg']
  cfg.l2norm = kwargs['l2norm']
  cfg.dim_ft = kwargs['dim_ft']
  cfg.dim_caption = kwargs['dim_caption']
  cfg.dim_joint_embeds = kwargs['dim_joint_embeds']

  cfg.max_words_in_caption = kwargs['max_words_in_caption']
  cfg.pool = kwargs['pool']

  adv_cfg = cfg.subcfgs[ADV]
  adv_cfg.dim_inputs = kwargs['dim_joint_embeds']
  adv_cfg.dim_hidden = kwargs['dim_hidden']
  adv_cfg.lr_mult = 5.

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
    CORR = 'corr'

  def _set_submods(self):
    return {
      ADV: adv.transform.Discriminator(self._config.subcfgs[ADV]),
    }

  def _add_input_in_mode(self, mode):
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, self._config.dim_ft), name='fts')
      captionids = tf.placeholder(
        tf.float32, shape=(None, self._config.max_words_in_caption, self._config.dim_caption), name='captionids')
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
      self.caption_pca_Ws = []
      self.caption_pca_Bs = []
      for g, dim_joint_embed in enumerate(self._config.dim_joint_embeds):
        caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W_%d'%g,
          shape=(self._config.dim_caption, dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        caption_pca_B = tf.contrib.framework.model_variable('caption_pca_B_%d'%g,
          shape=(dim_joint_embed,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._weights.append(caption_pca_W)
        self._weights.append(caption_pca_B)
        self.caption_pca_Ws.append(caption_pca_W)
        self.caption_pca_Bs.append(caption_pca_B)

      self.ft_pca_Ws = []
      self.ft_pca_Bs = []
      for g, dim_joint_embed in enumerate(self._config.dim_joint_embeds):
        ft_pca_W = tf.contrib.framework.model_variable('ft_pca_W_%d'%g,
          shape=(self._config.dim_ft, dim_joint_embed), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        ft_pca_B = tf.contrib.framework.model_variable('ft_pca_B_%d'%g,
          shape=(dim_joint_embed,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self._weights.append(ft_pca_W)
        self._weights.append(ft_pca_B)
        self.ft_pca_Ws.append(ft_pca_W)
        self.ft_pca_Bs.append(ft_pca_B)

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    with tf.variable_scope(self.name_scope):
      caption = in_ops[self.InKey.CAPTIONID]
      mask = in_ops[self.InKey.CAPTION_MASK]
      mask = tf.expand_dims(tf.to_float(mask), 2)
      caption_embeds = []
      for caption_pca_W, caption_pca_B, dim_joint_embed in zip(self.caption_pca_Ws, self.caption_pca_Bs, self._config.dim_joint_embeds):
        # caption_embed = tf.nn.conv1d(caption, tf.expand_dims(caption_pca_W, 0), 1, 'VALID')
        caption = tf.reshape(caption, (-1, self._config.dim_caption))
        caption_embed = tf.nn.xw_plus_b(caption, caption_pca_W, caption_pca_B)
        caption_embed = tf.reshape(caption_embed, (-1, self._config.max_words_in_caption, dim_joint_embed))
        caption_embed = tf.nn.tanh(caption_embed)
        if self._config.pool == 'mean':
          caption_embed = tf.reduce_sum(caption_embed*mask, 1) / tf.reduce_sum(mask, 1)
        else:
          caption_embed += 10.
          caption_embed = tf.reduce_max(caption_embed * mask, 1)
          caption_embed -= 10.
        if self._config.l2norm:
          caption_embed = tf.nn.l2_normalize(caption_embed, 1)
        caption_embeds.append(caption_embed)

      ft_embeds = []
      for ft_pca_W, ft_pca_B in zip(self.ft_pca_Ws, self.ft_pca_Bs):
        ft_embed = tf.nn.xw_plus_b(in_ops[self.InKey.FT], ft_pca_W, ft_pca_B)
        ft_embed = tf.nn.tanh(ft_embed)
        if self._config.l2norm:
          ft_embed = tf.nn.l2_normalize(ft_embed, 1)
        ft_embeds.append(ft_embed)

    def trn(ft_embeds, caption_embeds):
      # diversity 
      outputs = self.submods[ADV].get_out_ops_in_mode({
        self.submods[ADV].InKey.FT: caption_embeds
        }, None)
      txt_corr = outputs[self.submods[ADV].OutKey.CORR]

      outputs = self.submods[ADV].get_out_ops_in_mode({
        self.submods[ADV].InKey.FT: ft_embeds
        }, None)
      ft_corr = outputs[self.submods[ADV].OutKey.CORR]

      with tf.variable_scope(self.name_scope):
        corr = txt_corr + ft_corr

        # discriminty
        pos_ft_embeds = [ft_embed[:-self._config.num_neg] for ft_embed in ft_embeds]
        pos_caption_embeds = [caption_embed[:-self._config.num_neg] for caption_embed in caption_embeds]
        neg_ft_embeds = [ft_embed[-self._config.num_neg:] for ft_embed in ft_embeds]
        neg_caption_embeds = [caption_embed[-self._config.num_neg:] for caption_embed in caption_embeds]

        pos_sims = []
        for pos_ft_embed, pos_caption_embed in zip(pos_ft_embeds, pos_caption_embeds):
          pos_sim = tf.reduce_sum(pos_ft_embed * pos_caption_embed, 1)
          pos_sims.append(pos_sim)

        neg_caption_sims = []
        for pos_ft_embed, neg_caption_embed in zip(pos_ft_embeds, neg_caption_embeds):
          neg_caption_sim = tf.matmul(pos_ft_embed, neg_caption_embed, transpose_b=True)
          neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1) / 100.
          neg_caption_sims.append(neg_caption_sim)

        neg_ft_sims = []
        for neg_ft_embed, pos_caption_embed in zip(neg_ft_embeds, pos_caption_embeds):
          neg_ft_sim = tf.matmul(pos_caption_embed, neg_ft_embed, transpose_b=True)
          neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1) / 100.
          neg_ft_sims.append(neg_ft_sim)

        return corr, pos_sims, neg_caption_sims, neg_ft_sims

    def tst(ft_embeds, caption_embeds):
      with tf.variable_scope(self.name_scope):
        ft_embed = tf.concat(ft_embeds, 1)
        caption_embed = tf.concat(caption_embeds, 1)
        sim = tf.matmul(ft_embed, caption_embed, transpose_b=True)
        return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      corr, pos_sims, neg_caption_sims, neg_ft_sims = trn(ft_embeds, caption_embeds)
      sim = tst(ft_embeds, caption_embeds)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sims,
        self.OutKey.NF_SIM: neg_ft_sims,
        self.OutKey.NC_SIM: neg_caption_sims,
        self.OutKey.CORR: corr,
      }
    else:
      sim = tst(ft_embeds, caption_embeds)
      return {
        self.OutKey.SIM: sim,
      }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      corr = self._outputs[self.OutKey.CORR]
      self.op2monitor['corr_loss'] = corr

      pos_sims = self._outputs[self.OutKey.P_SIM]
      neg_caption_sims = self._outputs[self.OutKey.NC_SIM]
      neg_ft_sims = self._outputs[self.OutKey.NF_SIM]

      losses = []
      g = 0
      for pos_sim, neg_caption_sim, neg_ft_sim in zip(pos_sims, neg_caption_sims, neg_ft_sims):
        contrast_caption_loss = neg_caption_sim + self._config.margin - pos_sim
        contrast_caption_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_caption_loss))
        self.op2monitor['contrast_caption_loss_%d'%g] = tf.reduce_sum(contrast_caption_loss)

        contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
        contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))
        self.op2monitor['contrast_ft_loss_%d'%g] = tf.reduce_sum(contrast_ft_loss)

        loss = self._config.alpha * contrast_caption_loss + (1.0 - self._config.alpha) * contrast_ft_loss
        loss = tf.reduce_sum(loss)
        losses.append(loss)

        g += 1
      loss = tf.reduce_mean(losses)
      loss -= self._config.beta * corr
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

TrnReader = trntst_util.FreezeTrnReader
ValReader = trntst_util.FreezeValReader
TstReader = trntst_util.FreezeTstReader

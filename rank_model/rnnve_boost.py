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
import rnnve_orth

WE = rnnve_orth.WE
RNN = rnnve_orth.RNN
CELL = rnnve_orth.CELL
RCELL = rnnve_orth.RCELL


ModelConfig = rnnve_orth.ModelConfig
gen_cfg = rnnve_orth.gen_cfg


class Model(rnnve_orth.Model):
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
      caption = out_ops[rnn.OutKey.OUTPUT]
      mask = in_ops[self.InKey.CAPTION_MASK]
      mask = tf.expand_dims(tf.to_float(mask), 2)
      caption_embeds = []
      for caption_pca_W, caption_pca_B in zip(self.caption_pca_Ws, self.caption_pca_Bs):
        caption_embed = tf.nn.conv1d(caption, tf.expand_dims(caption_pca_W, 0), 1, 'VALID')
        caption_embed = tf.nn.tanh(caption_embed)
        if self._config.pool == 'mean':
          caption_embed = tf.reduce_sum(caption_embed*mask, 1) / tf.reduce_sum(mask, 1)
        else:
          _mask = tf.cast(mask, tf.bool)
          _mask = tf.tile(_mask, [1, 1, tf.shape(caption_embed)[-1]])
          caption_embed = tf.where(_mask, caption_embed, -10*tf.ones_like(caption_embed, dtype=tf.float32))
          caption_embed = tf.reduce_max(caption_embed, 1)
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
      with tf.variable_scope(self.name_scope):
        if self._config.loss == 'orth':
          dim_embed = sum(self._config.dim_joint_embeds)

          ft_embed = tf.concat(ft_embeds, 1)
          ft_corr = tf.square(tf.matmul(tf.transpose(ft_embed), ft_embed))
          ft_corr_sum = 0.
          base = 0
          total = 0
          for dim_joint_embed in self._config.dim_joint_embeds:
            ft_corr_sum += tf.reduce_sum(ft_corr[base:base+dim_joint_embed, base+dim_joint_embed:])
            base += dim_joint_embed
            total += dim_joint_embed * dim_joint_embed
          ft_corr_sum /= dim_embed * dim_embed - total

          caption_embed = tf.concat(caption_embeds, 1)
          caption_corr = tf.square(tf.matmul(tf.transpose(caption_embed), caption_embed))
          caption_corr_sum = 0.
          base = 0
          total = 0
          for dim_joint_embed in self._config.dim_joint_embeds:
            caption_corr_sum += tf.reduce_sum(caption_corr[base:base+dim_joint_embed, base+dim_joint_embed:])
            base += dim_joint_embed
            total += dim_joint_embed * dim_joint_embed
          caption_corr_sum /= dim_embed * dim_embed - total

          corr = ft_corr_sum + caption_corr_sum

          return corr
        else:
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
            # neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1) / 100.
            neg_caption_sims.append(neg_caption_sim)

          neg_ft_sims = []
          for neg_ft_embed, pos_caption_embed in zip(neg_ft_embeds, pos_caption_embeds):
            neg_ft_sim = tf.matmul(pos_caption_embed, neg_ft_embed, transpose_b=True)
            # neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1) / 100.
            neg_ft_sims.append(neg_ft_sim)

          return pos_sims, neg_caption_sims, neg_ft_sims

    def tst(ft_embeds, caption_embeds):
      with tf.variable_scope(self.name_scope):
        ft_embed = tf.concat(ft_embeds, 1)
        caption_embed = tf.concat(caption_embeds, 1)
        if self._config.loss == 'orth':
          dim_embed = sum(self._config.dim_joint_embeds)
          ft_corr = tf.square(tf.matmul(tf.transpose(ft_embed), ft_embed))
          ft_corr_sum = 0.
          base = 0
          total = 0
          for dim_joint_embed in self._config.dim_joint_embeds:
            ft_corr_sum += tf.reduce_sum(ft_corr[base:base+dim_joint_embed, base+dim_joint_embed:])
            base += dim_joint_embed
            total += dim_joint_embed * dim_joint_embed
          ft_corr_sum /= dim_embed * dim_embed - total

          caption_corr = tf.square(tf.matmul(tf.transpose(caption_embed), caption_embed))
          caption_corr_sum = 0.
          base = 0
          total = 0
          for dim_joint_embed in self._config.dim_joint_embeds:
            caption_corr_sum += tf.reduce_sum(caption_corr[base:base+dim_joint_embed, base+dim_joint_embed:])
            base += dim_joint_embed
            total += dim_joint_embed * dim_joint_embed
          caption_corr_sum /= dim_embed * dim_embed - total

          corr = ft_corr_sum + caption_corr_sum
          return corr
        else:
          sim = tf.matmul(ft_embed, caption_embed, transpose_b=True)
          return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      if self._config.loss == 'orth':
        corr = trn(ft_embeds, caption_embeds)
        tst_corr = tst(ft_embeds, caption_embeds)
        return {
          self.OutKey.CORR: corr,
          self.OutKey.SIM: tst_corr,
        }
      else:
        pos_sims, neg_caption_sims, neg_ft_sims = trn(ft_embeds, caption_embeds)
        sim = tst(ft_embeds, caption_embeds)
        return {
          self.OutKey.SIM: sim,
          self.OutKey.P_SIM: pos_sims,
          self.OutKey.NF_SIM: neg_ft_sims,
          self.OutKey.NC_SIM: neg_caption_sims,
        }
    else:
      sim = tst(ft_embeds, caption_embeds)
      return {
        self.OutKey.SIM: sim,
      }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      if self._config.loss == 'orth':
        loss = self._outputs[self.OutKey.CORR]
        self.op2monitor['loss'] = loss
      else:
        pos_sims = self._outputs[self.OutKey.P_SIM]
        neg_caption_sims = self._outputs[self.OutKey.NC_SIM]
        neg_ft_sims = self._outputs[self.OutKey.NF_SIM]

        w = tf.ones((tf.shape(pos_sims[0])[0],))

        loss = 0.
        for g in range(len(pos_sims)):
          nu = 2. / (2. + g)
          if g == 0:
            pos_sim = pos_sims[g]
            neg_caption_sim = neg_caption_sims[g]
            neg_ft_sim = neg_ft_sims[g]
          else:
            # pos_sim = tf.stop_gradient(pos_sim)
            # neg_caption_sim = tf.stop_gradient(neg_caption_sim)
            # neg_ft_sim = tf.stop_gradient(neg_ft_sim)

            pos_sim = (1.-nu) * pos_sim + nu * pos_sims[g]
            neg_caption_sim = (1.-nu) * neg_caption_sim + nu * neg_caption_sims[g]
            neg_ft_sim = (1.-nu) * neg_ft_sim + nu * neg_ft_sims[g]
          max_neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1)/100.
          max_neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1)/100.

          contrast_caption_loss = max_neg_caption_sim + self._config.margin - pos_sim
          contrast_caption_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_caption_loss))
          self.op2monitor['contrast_caption_loss_%d'%g] = tf.reduce_sum(contrast_caption_loss)

          contrast_ft_loss = max_neg_ft_sim + self._config.margin - pos_sim
          contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))
          self.op2monitor['contrast_ft_loss_%d'%g] = tf.reduce_sum(contrast_ft_loss)

          group_loss = self._config.alpha * contrast_caption_loss + (1. - self._config.alpha) * contrast_ft_loss
          loss += tf.reduce_sum(w * group_loss)
          self.op2monitor['loss_%d'%g] = tf.reduce_sum(group_loss)

          w = self._config.alpha * tf.to_float(contrast_caption_loss > 0) + (1. - self._config.alpha) * tf.to_float(contrast_ft_loss > 0)
          w = tf.stop_gradient(w)
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
# OrthTrnTst = rnnve_orth.OrthTrnTst


TrnReader = trntst_util.TrnReader
ValReader = trntst_util.ValReader
TstReader = trntst_util.TstReader
# OrthReader = rnnve_orth.OrthReader

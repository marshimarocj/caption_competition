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
import rnnve

WE = rnnve.WE
RNN = rnnve.RNN
CELL = rnnve.CELL
RCELL = rnnve.RCELL

ModelConfig = rnnve.ModelConfig
gen_cfg = rnnve.gen_cfg


class Model(rnnve.Model):
  class OutKey(enum.Enum):
    CAPTION_EMBED = 'caption_embed'
    FT_EMBED = 'ft_embed'

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

    if mode == framework.model.module.Mode.TST:
      return {
        self.OutKey.FT_EMBED: ft_embed,
        self.OutKey.CAPTION_EMBED: caption_embed,
      }

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.CAPTION_EMBED: self._outputs[self.OutKey.CAPTION_EMBED],
      self.OutKey.FT_EMBED: self._outputs[self.OutKey.FT_EMBED],
    }


PathCfg = rnnve.PathCfg


class TrnTst(framework.model.trntst.TrnTst):
  def predict_in_tst(self, sess, tst_reader, predict_file):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_tst()
    caption_embeds = []
    ft_embeds = []
    for data in tst_reader.yield_tst_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      caption_embed, ft_embed = sess.run(
        [op_dict[self.model.OutKey.CAPTION_EMBED], op_dict[self.model.OutKey.FT_EMBED]], feed_dict=feed_dict)
      caption_embeds.append(caption_embed)
      ft_embeds.append(ft_embed)
    caption_embeds = np.concatenate(caption_embeds, 0)
    ft_embeds = np.concatenate(ft_embeds, 0)
    np.savez_compressed(predict_file, caption_embeds=caption_embeds, ft_embeds=ft_embeds)


class TstReader(framework.model.data.Reader):
  def __init__(self, ft_files, annotation_file):
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

  def yield_tst_batch(self, batch_size):
    num = self.fts.shape[0]
    for i in range(0, num, batch_size):
      fts = self.fts[i:i+batch_size]
      captionids = self.captionids[i:i+batch_size]
      caption_masks = self.caption_masks[i:i+batch_size]

      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
      }

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
    CAPTION_OUTPUT = 'caption_output'

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

    caption_embed = out_ops[rnn.OutKey.OUTPUT]
    if mode == framework.model.module.Mode.TST:
      return {
        self.OutKey.CAPTION_OUTPUT: caption_embed,
      }

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.CAPTION_OUTPUT: self._outputs[self.OutKey.CAPTION_OUTPUT],
    }


PathCfg = rnnve.PathCfg


class TrnTst(framework.model.trntst.TrnTst):
  def predict_in_tst(self, sess, tst_reader, predict_file):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_tst()
    caption_outputs = []
    caption_masks = []
    ft_idxs = []
    for data in tst_reader.yield_tst_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      caption_output = sess.run(
        op_dict[self.model.OutKey.CAPTION_OUTPUT], feed_dict=feed_dict)
      caption_outputs.append(caption_output)
      caption_masks.append(data['caption_masks'])
      ft_idxs.append(data['ft_idxs'])
    caption_outputs = np.concatenate(caption_outputs, 0)
    caption_masks = np.concatenate(caption_masks, 0)
    ft_idxs = np.concatenate(ft_idxs, 0)
    with open(predict_file, 'w') as fout:
      cPickle.dump([ft_idxs, caption_outputs, caption_masks], fout)


class TstReader(framework.model.data.Reader):
  def __init__(self, annotation_file):
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

  def yield_tst_batch(self, batch_size):
    num = self.ft_idxs.shape[0]
    for i in range(0, num, batch_size):
      ft_idxs = self.ft_idxs[i:i+batch_size]
      captionids = self.captionids[i:i+batch_size]
      caption_masks = self.caption_masks[i:i+batch_size]

      yield {
        'ft_idxs': ft_idxs,
        'captionids': captionids,
        'caption_masks': caption_masks,
      }

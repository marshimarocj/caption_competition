import enum
import sys
import os
import cPickle
import json
import random
sys.path.append('../')

import numpy as np
import tensorflow as tf

import framework.model.module
import framework.model.trntst
import framework.model.data
import framework.util.caption.utility
import framework.impl.encoder.pca
import decoder.rnn
import trntst_util

VE = 'encoder'
VD = 'decoder'
CELL = decoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.pca.Config()
    self.subcfgs[VD] = decoder.rnn.Config()

    self.search_strategy = 'beam'
    self.tst_num_sample = -1
    self.tst_sample_topk= -1

  def _assert(self):
    assert self.subcfgs[VE].dim_output == self.subcfgs[VD].subcfgs[CELL].dim_hidden


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.val_loss = False
  cfg.monitor_iter = 100
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 64
  cfg.base_lr = 1e-4
  cfg.num_epoch = kwargs['num_epoch']

  enc = cfg.subcfgs[VE]
  enc.dim_ft = kwargs['dim_ft']
  enc.dim_output = kwargs['dim_hidden']
  # enc.keepin_prob = kwargs['content_keepin_prob']

  dec = cfg.subcfgs[VD]
  dec.num_step = kwargs['num_step']
  dec.dim_input = kwargs['dim_input']
  dec.dim_hidden = kwargs['dim_hidden']

  cell = dec.subcfgs[CELL]
  cell.dim_hidden = kwargs['dim_hidden']
  cell.dim_input = kwargs['dim_input']
  cell.keepout_prob = kwargs['cell_keepout_prob']
  cell.keepin_prob = kwargs['cell_keepin_prob']

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'vevd.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    CAPTIONID = 'captionids'
    CAPTION_MASK = 'caption_masks'
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    LOGIT = 'logit'
    OUT_WID = 'out_wid'
    LOG_PROB = 'log_prob'
    BEAM_CUM_LOG_PROB = 'beam_cum_log_prob'
    BEAM_PRE = 'beam_pre'
    BEAM_END = 'beam_end'

  def _set_submods(self):
    return {
      VE: framework.impl.encoder.pca.Encoder(self._config.subcfgs[VE]),
      VD: decoder.rnn.Decoder(self._config.subcfgs[VD]),
    }

  def _add_input_in_mode(self, mode):
    if mode == framework.model.module.Mode.TRN_VAL:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, self._config.subcfgs[VE].dim_ft), name=self.InKey.FT.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)
        # trn only
        captionids = tf.placeholder(
          tf.int32, shape=(None, self._config.subcfgs[VD].num_step), name=self.InKey.CAPTIONID.value)
        caption_masks = tf.placeholder(
          tf.float32, shape=(None, self.config.subcfgs[VD].num_step), name=self.InKey.CAPTION_MASK.value)

        out = {
          self.InKey.FT: fts,
          self.InKey.IS_TRN: is_training,
          self.InKey.CAPTIONID: captionids,
          self.InKey.CAPTION_MASK: caption_masks,
        }
      return out
    else:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, self._config.subcfgs[VE].dim_ft), name=self.InKey.FT.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)

      return {
        self.InKey.FT: fts,
        self.InKey.IS_TRN: is_training,
      }

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[VE]
    decoder = self.submods[VD]

    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.FT: in_ops[self.InKey.FT],
    }, mode)
    ft_embed = out_ops[encoder.OutKey.EMBED] # (None, dim_output)

    def trn_val(ft_embed):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(ft_embed)[0]
        init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      caption_masks = in_ops[self.InKey.CAPTION_MASK]

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: in_ops[self.InKey.CAPTIONID],
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, is_trn=True)
      out_wid = out_ops[decoder.OutKey.OUT_WID]
      logit = out_ops[decoder.OutKey.LOGIT]

      return {
        self.OutKey.OUT_WID: out_wid,
        self.OutKey.LOGIT: logit,
      }

    def tst(ft_embed):
      with tf.variable_scope(self.name_scope):
        batch_size = tf.shape(ft_embed)[0]
        init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
      }
      if self._config.search_strategy == 'beam':
        out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
          task='generation', search_strategy='beam')
        return {
          self.OutKey.OUT_WID: out_ops[decoder.OutKey.OUT_WID],
          self.OutKey.BEAM_CUM_LOG_PROB: out_ops[decoder.OutKey.BEAM_CUM_LOG_PROB],
          self.OutKey.BEAM_PRE: out_ops[decoder.OutKey.BEAM_PRE],
          self.OutKey.BEAM_END: out_ops[decoder.OutKey.BEAM_END],
        }
      elif self._config.search_strategy == 'sample':
        out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
          search_strategy='sample', num_sample=self._config.tst_num_sample, topk=self._config.tst_sample_topk, task='generation')
        return {
          self.OutKey.OUT_WID: out_ops[decoder.OutKey.OUT_WID],
          self.OutKey.LOG_PROB: out_ops[decoder.OutKey.LOG_PROB],
        }

    delegate = {
      framework.model.module.Mode.TRN_VAL: trn_val,
      framework.model.module.Mode.TST: tst,
    }
    return delegate[mode](ft_embed)

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      logits = self._outputs[self.OutKey.LOGIT] # (None*num_step, num_words)
      xentropy_loss = framework.util.expanded_op.cross_entropy_loss_on_rnn_logits(
        self._inputs[self.InKey.CAPTIONID], self._inputs[self.InKey.CAPTION_MASK], logits)
      self.op2monitor['loss'] = xentropy_loss
      loss_op = xentropy_loss

    return loss_op

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
    }

  def op_in_tst(self):
    if self._config.search_strategy == 'beam':
      return {
        self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
        self.OutKey.BEAM_CUM_LOG_PROB: self._outputs[self.OutKey.BEAM_CUM_LOG_PROB],
        self.OutKey.BEAM_PRE: self._outputs[self.OutKey.BEAM_PRE],
        self.OutKey.BEAM_END: self._outputs[self.OutKey.BEAM_END],
      }
    elif self._config.search_strategy == 'sample':
      return {
        self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
        self.OutKey.LOG_PROB: self._outputs[self.OutKey.LOG_PROB],
      }


PathCfg = trntst_util.PathCfg


class TrnTst(framework.model.trntst.TrnTst):
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    framework.model.trntst.TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = framework.util.caption.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

  def _construct_feed_dict_in_trn(self, data):
    fts = data['fts']
    captionids = data['captionids']
    caption_masks = data['caption_masks']

    return {
      self.model.inputs[self.model.InKey.FT]: fts,
      self.model.inputs[self.model.InKey.CAPTIONID]: captionids,
      self.model.inputs[self.model.InKey.CAPTION_MASK]: caption_masks,
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    trntst_util.predict_and_eval_in_val(self, sess, tst_reader, metrics)

  def predict_in_tst(self, sess, tst_reader, predict_file):
    trntst_util.predict_in_tst(self, sess, tst_reader, predict_file, 
      self.model_cfg.search_strategy)


Reader = trntst_util.Reader

import enum
import sys
import os
import cPickle
import json
import random
sys.path.append('../')
import time

import tensorflow as tf
import numpy as np

import framework.model.module
import framework.model.trntst
import framework.model.data
import framework.util.caption.utility
import framework.impl.encoder.dnn
import decoder.att_rnn
import trntst_util


VE = 'encoder'
AD = 'decoder'
CELL = decoder.att_rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.dnn.Config()
    self.subcfgs[AD] = decoder.att_rnn.Config()

    self.search_strategy = 'beam'

  def _assert(self):
    assert self.subcfgs[VE].dim_output == self.subcfgs[AD].subcfgs[CELL].dim_hidden


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
  enc.dim_fts = kwargs['dim_fts']
  enc.dim_output = kwargs['dim_hidden']
  enc.keepin_prob = kwargs['content_keepin_prob']

  dec = cfg.subcfgs[AD]
  dec.num_step = kwargs['num_step']
  dec.dim_input = kwargs['dim_input']
  dec.dim_hidden = kwargs['dim_hidden']
  dec.dim_attention = kwargs['dim_attention']
  dec.num_ft = kwargs['num_ft']
  dec.dim_ft = kwargs['dim_ft']

  cell = dec.subcfgs[CELL]
  cell.dim_hidden = kwargs['dim_hidden']
  cell.dim_input = kwargs['dim_input']
  cell.keepout_prob = kwargs['cell_keepout_prob']
  cell.keepin_prob = kwargs['cell_keepin_prob']

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'vead.Model'

  class InKey(enum.Enum):
    FT = 'fts' # (None, num_ft, dim_ft)
    FT_MASK = 'ft_masks' # (None, num_ft)
    CAPTIONID = 'captionids' # (None, max_caption_len)
    CAPTION_MASK = 'caption_masks' # (None, max_caption_len)
    INIT_WID = 'init_wid'
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    LOGIT = 'logit'
    OUT_WID = 'out_wid'
    BEAM_CUM_LOG_PROB = 'beam_cum_log_prob'
    BEAM_PRE = 'beam_pre'
    BEAM_END = 'beam_end'

  def _set_submods(self):
    return {
      VE: framework.impl.encoder.dnn.Encoder(self._config.subcfgs[VE]),
      AD: decoder.attention.Decoder(self._config.subcfgs[AD]),
    }

  def _add_input_in_mode(self, mode):
    encoder_cfg = self._config.subcfgs[VE]
    decoder_cfg = self._config.subcfgs[AD]
    if mode == framework.model.module.Mode.TRN_VAL:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, decoder_cfg.num_ft, sum(encoder_cfg.dim_fts)), name=self.InKey.FT.value)
        ft_masks = tf.placeholder(
          tf.float32, shape=(None, decoder_cfg.num_ft), name=self.InKey.FT_MASK.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)
        # trn only
        captionids = tf.placeholder(
          tf.int32, shape=(None, decoder_cfg.max_words_in_caption), name=self.InKey.CAPTIONID.value)
        caption_masks = tf.placeholder(
          tf.float32, shape=(None, decoder_cfg.max_words_in_caption), name=self.InKey.CAPTION_MASK.value)
        # tst only
        init_wordids = tf.placeholder(
          tf.int32, shape=(None,), name=self.InKey.INIT_WID.value)

        return {
          self.InKey.FT: fts,
          self.InKey.FT_MASK: ft_masks,
          self.InKey.CAPTIONID: captionids,
          self.InKey.CAPTION_MASK: caption_masks,
          self.InKey.INIT_WID: init_wordids,
          self.InKey.IS_TRN: is_training,
        }
    else:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, decoder_cfg.num_ft, sum(encoder_cfg.dim_fts)), name=self.InKey.FT.value)
        ft_masks = tf.placeholder(
          tf.float32, shape=(None, decoder_cfg.num_ft), name=self.InKey.FT_MASK.value)
        init_wordids = tf.placeholder(
          tf.int32, shape=(None, ), name=self.InKey.INIT_WID.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)

        return {
          self.InKey.FT: fts,
          self.InKey.FT_MASK: ft_masks,
          self.InKey.INIT_WID: init_wordids,
          self.InKey.IS_TRN: is_training,
        }

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode):
    fts = in_ops[self.InKey.FT]
    ft_masks = in_ops[self.InKey.FT_MASK]
    with tf.variable_scope(self.name_scope):
      ft = fts[:, 0]
    out_ops = self.submods[VE].get_out_ops_in_mode({
      self.submods[VE].InKey.FT: ft,
      self.submods[VE].InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
    }, mode)
    ft_embed = out_ops[self.submods[VE].OutKey.EMBED]

    ad_inputs = {
      self.submods[AD].InKey.INIT_FT: ft_embed,
      self.submods[AD].InKey.FT: fts,
      self.submods[AD].InKey.FT_MASK: ft_masks,
      self.submods[AD].InKey.INIT_WID: in_ops[self.InKey.INIT_WID],
    }
    if mode == framework.model.module.Mode.TRN_VAL:
      ad_inputs.update({
        self.submods[AD].InKey.CAPTIONID: in_ops[self.InKey.CAPTIONID],
      })

    out_ops = self.submods[AD].get_out_ops_in_mode(ad_inputs, mode)

    out = {
      self.OutKey.OUT_WID: out_ops[self.submods[AD].OutKey.OUT_WID],
    }
    if mode == framework.model.module.Mode.TRN_VAL:
      out.update({
        self.OutKey.LOGIT: out_ops[self.submods[AD].OutKey.LOGIT],
      })
    else:
      out.update({
        self.OutKey.BEAM_CUM_LOG_PROB: out_ops[self.submods[AD].OutKey.BEAM_CUM_LOG_PROB],
        self.OutKey.BEAM_PRE: out_ops[self.submods[AD].OutKey.BEAM_PRE],
        self.OutKey.BEAM_END: out_ops[self.submods[AD].OutKey.BEAM_END],
      })

    return out

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      logits = self._outputs[self.OutKey.LOGIT]
      loss_op = framework.util.expanded_op.cross_entropy_loss_on_rnn_logits(
        self._inputs[self.InKey.CAPTIONID], self._inputs[self.InKey.CAPTION_MASK], logits)
      self.op2monitor['loss'] = loss_op

    return loss_op

  def op_in_tst(self):
    op_dict = {
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
    }
    if not self._config.subcfgs[AD].greedy_or_beam:
      op_dict.update({
        self.OutKey.BEAM_CUM_LOG_PROB: self._outputs[self.OutKey.BEAM_CUM_LOG_PROB],
        self.OutKey.BEAM_PRE: self._outputs[self.OutKey.BEAM_PRE],
        self.OutKey.BEAM_END: self._outputs[self.OutKey.BEAM_END],
      })
    return op_dict

  def op_in_val(self):
    op_dict = framework.model.module.AbstractModel.op_in_val(self)
    op_dict.update({
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
    })
    return op_dict


PathCfg = trntst_util.AttPathCfg


class TrnTst(framework.model.trntst.TrnTst):
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    framework.model.trntst.TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = framework.util.caption.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

  def _construct_feed_dict_in_trn(self, data):
    fts = data['fts']
    ft_masks = data['ft_masks']
    captionids = data['captionids']
    caption_masks = data['caption_masks']

    return {
      self.model.inputs[self.model.InKey.FT]: fts,
      self.model.inputs[self.model.InKey.FT_MASK]: ft_masks,
      self.model.inputs[self.model.InKey.CAPTIONID]: captionids,
      self.model.inputs[self.model.InKey.CAPTION_MASK]: caption_masks,
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    trntst_util.predict_and_eval_in_val(self, sess, tst_reader, metrics, att=True)

  def predict_in_tst(self, sess, tst_reader, predict_file):
    trntst_util.predict_in_tst(self, sess, tst_reader, predict_file, 
      self.model_cfg.search_strategy, att=True)


Reader = trntst_util.AttReader

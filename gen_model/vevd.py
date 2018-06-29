import enum
import sys
import os
import cPickle
import json
import random
sys.path.append('../')

import numpy as np
import tensorflow as tf

from bleu import bleu
from cider import cider

import framework.model.module
import framework.model.trntst
import framework.model.data
import framework.util.caption.utility
import framework.impl.encoder.dnn
import decoder.rnn
import trntst_util

VE = 'encoder'
VD = 'decoder'
CELL = decoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.dnn.Config()
    self.subcfgs[VD] = decoder.rnn.Config()

    self.search_strategy = 'beam'

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
  enc.dim_fts = kwargs['dim_fts']
  enc.dim_output = kwargs['dim_hidden']
  enc.keepin_prob = kwargs['content_keepin_prob']

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
      VE: framework.impl.encoder.dnn.Encoder(self._config.subcfgs[VE]),
      VD: decoder.rnn.Decoder(self._config.subcfgs[VD]),
    }

  def _add_input_in_mode(self, mode):
    if mode == framework.model.module.Mode.TRN_VAL:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(self._config.subcfgs[VE].dim_fts)), name=self.InKey.FT.value)
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
          tf.float32, shape=(None, sum(self._config.subcfgs[VE].dim_fts)), name=self.InKey.FT.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)

      return {
        self.InKey.FT: fts,
        self.InKey.CAPTIONID: captionids,
        self.InKey.CAPTION_MASK: caption_masks,
        self.InKey.IS_TRN: is_training,
      }

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[VE]
    decoder = self.submods[VD]

    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.FT: in_ops[self.InKey.FT],
      encoder.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
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
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
        task='generation', strategy=self._config.strategy)
      return {
        self.OutKey.OUT_WID: out_ops[decoder.OutKey.OUT_WID],
        self.OutKey.BEAM_CUM_LOG_PROB: out_ops[decoder.OutKey.BEAM_CUM_LOG_PROB],
        self.OutKey.BEAM_PRE: out_ops[decoder.OutKey.BEAM_PRE],
        self.OutKey.BEAM_END: out_ops[decoder.OutKey.BEAM_END],
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
    if self._config.strategy == 'beam':
      return {
        self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
        self.OutKey.BEAM_CUM_LOG_PROB: self._outputs[self.OutKey.BEAM_CUM_LOG_PROB],
        self.OutKey.BEAM_PRE: self._outputs[self.OutKey.BEAM_PRE],
        self.OutKey.BEAM_END: self._outputs[self.OutKey.BEAM_END],
      }
    elif self._config.strategy == 'sample':
      return {
        self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
        self.OutKey.LOG_PROB: self._outputs[self.OutKey.LOG_PROB],
      }


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    framework.model.trntst.PathCfg.__init__(self)
    # manually provided in the cfg file
    self.split_dir = ''
    self.annotation_dir = ''
    self.output_dir = ''
    self.trn_ftfiles = []
    self.val_ftfiles = []
    self.tst_ftfiles = []

    # automatically generated paths
    self.trn_videoid_file = ''
    self.val_videoid_file = ''
    self.trn_annotation_file = ''
    self.val_annotation_file = ''
    self.groundtruth_file = ''
    self.word_file = ''


class TrnTst(trntst_util.TrnTst):
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


class Reader(framework.model.data.Reader):
  def __init__(self, ft_files, videoid_file, 
      shuffle=True, annotation_file=None, captionstr_file=None):
    self.fts = np.empty(0) # (numVideo, dimVideo)
    self.ft_idxs = np.empty(0) # (num_caption,)
    self.captionids = np.empty(0) # (num_caption, maxWordsInCaption)
    self.caption_masks = np.empty(0) # (num_caption, maxWordsInCaption)
    self.videoids = []
    self.videoid2captions = {} # (numVideo, numGroundtruth)

    self.shuffled_idxs = [] # (num_caption,)
    self.num_caption = 0 # used in trn and val
    self.num_ft = 0

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.fts = self.fts.astype(np.float32)
    self.num_ft = self.fts.shape[0]

    self.videoids = np.load(open(videoid_file))

    if annotation_file is not None:
      self.ft_idxs, self.captionids, self.caption_masks = cPickle.load(file(annotation_file))
      self.num_caption = self.ft_idxs.shape[0]
    if captionstr_file is not None:
      videoid2captions = cPickle.load(open(captionstr_file))
      for videoid in self.videoids:
        self.videoid2captions[videoid] = videoid2captions[videoid]

    self.shuffled_idxs = range(self.num_caption)
    if shuffle:
      random.shuffle(self.shuffled_idxs)

  def num_record(self):
    return self.num_caption

  def yield_trn_batch(self, batch_size, **kwargs):
    for i in range(0, self.num_caption, batch_size):
      start = i
      end = i + batch_size
      idxs = self.shuffled_idxs[start:end]

      yield {
        'fts': self.fts[self.ft_idxs[idxs]],
        'captionids': self.captionids[idxs],
        'caption_masks': self.caption_masks[idxs],
        'vids': self.videoids[self.ft_idxs[idxs]],
      }

  def yield_val_batch(self, batch_size, **kwargs):
    for i in range(0, self.num_caption, batch_size):
      start = i
      end = i + batch_size
      idxs = self.shuffled_idxs[start:end]

      yield {
        'fts': self.fts[self.ft_idxs[idxs]],
        'captionids': self.captionids[idxs],
        'caption_masks': self.caption_masks[idxs],
      }

  # when we generate tst batch, we never shuffle as we are not doing training
  def yield_tst_batch(self, batch_size, **kwargs):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size

      yield {
        'fts': self.fts[start:end],
      }

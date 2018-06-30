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
import framework.impl.encoder.dnn
import decoder.rnn
import trntst_util
import service.fast_cider


VE = 'encoder'
VD = 'decoder'
CELL = decoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.dnn.Config()
    self.subcfgs[VD] = decoder.rnn.Config()
    self.num_neg = 64
    self.max_margin = 0.5

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
  cfg.num_neg = kwargs['num_neg']
  cfg.max_margin = kwargs['max_margin']

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
    NCAPTIONID = 'neg_captionids'
    NCAPTION_MASK = 'neg_caption_masks'
    DELTA = 'delta'
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    LOG_PROB = 'log_prob'
    NLOG_PROB = 'neg_log_prob'

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
        neg_captionids = tf.placeholder(
          tf.int32, shape=(self._config.num_neg, self._config.subcfgs[VD].num_step), name=self.InKey.NCAPTIONID.value)
        neg_caption_masks = tf.placeholder(
          tf.float32, shape=(self._config.num_neg, self.config.subcfgs[VD].num_step), name=self.InKey.NCAPTION_MASK.value)
        deltas = tf.placeholder(
          tf.float32, shape=(None, self._config.num_neg), name=self.InKey.DELTA.value)

      return {
        self.InKey.FT: fts,
        self.InKey.IS_TRN: is_training,
        self.InKey.CAPTIONID: captionids,
        self.InKey.CAPTION_MASK: caption_masks,
        self.InKey.NCAPTIONID: neg_captionids,
        self.InKey.NCAPTION_MASK: neg_caption_masks,
        self.InKey.DELTA: deltas,
      }
    else:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(self._config.subcfgs[VE].dim_fts)), name=self.InKey.FT.value)
        captionids = tf.placeholder(
          tf.int32, shape=(None, self._config.subcfgs[VD].num_step), name=self.InKey.CAPTIONID.value)
        caption_masks = tf.placeholder(
          tf.float32, shape=(None, self._config.subcfgs[VD].num_step), name=self.InKey.CAPTION_MASK.value)
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
      batch_size = tf.shape(ft_embed)[0]

      # pos
      caption_masks = in_ops[self.InKey.CAPTION_MASK]
      init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: in_ops[self.InKey.CAPTIONID],
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, is_trn=False)
      log_prob = out_ops[decoder.OutKey.LOG_PROB]
      norm_log_prob = tf.reduce_sum(log_prob*caption_masks[:, 1:], axis=1) / \
        tf.reduce_sum(caption_masks[:, 1:], axis=1) # (None,)

      # neg
      ft_embed = tf.tile(tf.expand_dims(ft_embed, 1), [1, self._config.num_neg, 1]) # (None, num_neg, dim_output)
      ft_embed = tf.reshape(ft_embed, (-1, self._config.subcfgs[VE].dim_output)) # (None*num_neg, dim_output)
      neg_captionid = in_ops[self.InKey.NCAPTIONID]
      neg_captionid = tf.tile(tf.expand_dims(neg_captionid, 0), [batch_size, 1, 1]) # (None, num_neg, num_step)
      neg_captionid = tf.reshape(neg_captionid, (-1, self._config.subcfgs[VD].num_step))
      neg_caption_masks = in_ops[self.InKey.NCAPTION_MASK]
      neg_caption_masks = tf.tile(tf.expand_dims(neg_caption_masks, 0), [batch_size, 1, 1]) # (None, num_neg, num_step)
      neg_caption_masks = tf.reshape(neg_caption_masks, (-1, self._config.subcfgs[VD].num_step))
      init_wid = tf.zeros((batch_size*self._config.num_neg,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
        decoder.InKey.CAPTIONID: neg_captionid,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, is_trn=False)

      neg_log_prob = out_ops[decoder.OutKey.LOG_PROB]
      norm_neg_log_prob = tf.reduce_sum(neg_log_prob * neg_caption_masks[:, 1:], axis=1) / \
        tf.reduce_sum(neg_caption_masks[:, 1:], axis=1) # (None*num_neg,)
      norm_neg_log_prob = tf.reshape(norm_neg_log_prob, (-1, self._config.num_neg)) # (None, num_neg)

      return {
        self.OutKey.LOG_PROB: norm_log_prob,
        self.OutKey.NLOG_PROB: norm_neg_log_prob,
      }

    def tst(ft_embed):
      batch_size = tf.shape(ft_embed)[0]
      init_wid = tf.zeros((batch_size,), dtype=tf.int32)

      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.CAPTIONID: in_ops[self.InKey.CAPTIONID],
        decoder.InKey.INIT_WID: init_wid,
      }
      decoder.is_trn = False
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, task='retrieval')

      caption_masks = in_ops[self.InKey.CAPTION_MASK]
      log_prob = out_ops[decoder.OutKey.LOG_PROB]
      norm_log_prob = tf.reduce_sum(log_prob*caption_masks[:, 1:], axis=1) / \
        tf.reduce_sum(caption_masks[:, 1:], axis=1) # (None,)
      return {
        self.OutKey.LOG_PROB: norm_log_prob
      }

    delegate = {
      framework.model.module.Mode.TRN_VAL: trn_val,
      framework.model.module.Mode.TST: tst,
    }
    return delegate[mode](ft_embed)

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      log_prob = self._outputs[self.OutKey.LOG_PROB]
      log_prob = tf.expand_dims(log_prob, 1) # (None, 1)
      neg_log_prob = self._outputs[self.OutKey.NLOG_PROB] # (None, num_neg)

      deltas = self._inputs[self.InKey.DELTA]
      max_margin = self._config.max_margin * tf.ones_like(deltas, dtype=tf.float32)
      margin = tf.minimum(deltas, max_margin)
      loss_op = tf.reduce_logsumexp(100*(margin + neg_log_prob), axis=1) / 100.
      loss_op -= log_prob
      loss_op = tf.maximum(tf.zeros_like(loss_op), loss_op)
      loss_op = tf.reduce_mean(loss_op)
      self.op2monitor['loss'] = loss_op

    return loss_op

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.LOG_PROB: self._outputs[self.OutKey.LOG_PROB],
    }

  def op_in_tst(self):
    return {
      self.OutKey.LOG_PROB: self._outputs[self.OutKey.LOG_PROB],
    }


PathCfg = trntst_util.ScorePathCfg

class TrnTst(framework.model.trntst.TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
      self.model.inputs[self.model.InKey.NCAPTIONID]: data['neg_captionids'],
      self.model.inputs[self.model.InKey.NCAPTION_MASK]: data['neg_caption_masks'],
      self.model.inputs[self.model.InKey.DELTA]: data['deltas'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    mir = 0.
    num = 0
    for data in tst_reader.yield_val_batch(batch_size):
      feed_dict= {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sims = sess.run(op_dict[self.model.OutKey.LOG_PROB], feed_dict=feed_dict)
      idxs = np.argsort(-sims)
      rank = np.where(idxs == data['gt'])[0][0]
      rank += 1
      mir += 1. / rank
      num += 1
    mir /= num
    metrics['mir'] = mir

  def predict_in_tst(self, sess, tst_reader, predict_file):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    sims = []
    for data in tst_reader.yield_tst_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sim = sess.run(op_dict[self.model.OutKey.LOG_PROB], feed_dict=feed_dict)
      sims.append(sim)
    sims = np.array(sims, dtype=np.float32)
    np.save(predict_file, sims)


class TrnReader(framework.model.data.Reader):
  def __init__(self, num_neg, ft_files, annotation_file, vid_file, word_file, gt_file):
    self.num_neg = num_neg
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.idxs = []
    self.vids = np.empty(0)
    self.num_caption = 0

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

    self.vids = np.load(vid_file)
    self.int2str = framework.util.caption.utility.CaptionInt2str(word_file)

    self.num_caption = self.captionids.shape[0]
    self.idxs = range(self.num_caption)

    with open(gt_file) as f:
      vid2captions = cPickle.load(f)
    self.cider_scorer = service.fast_cider.CiderScorer()
    self.cider_scorer.init_refs(vid2captions)

  def num_record(self):
    return self.num_caption

  def reset(self):
    random.shuffle(self.idxs)

  def yield_trn_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      pos_idxs = self.idxs[i:i+batch_size]
      pos_ft_idxs = set(self.ft_idxs[pos_idxs].tolist())

      pos_fts = self.fts[self.ft_idxs[pos_idxs]]
      pos_captionids = self.captionids[pos_idxs]
      pos_caption_masks = self.caption_masks[pos_idxs]
      pos_vids = self.vids[self.ft_idxs[pos_idxs]]

      idxs = range(self.num_caption)
      random.shuffle(idxs)

      neg_captionids= []
      neg_caption_masks = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        if ft_idx not in pos_ft_idxs:
          neg_captionids.append(self.captionids[idx])
          neg_caption_masks.append(self.caption_masks[idx])
          if len(neg_captionids) == self.num_neg:
            break
      neg_captionids = np.array(neg_captionids, dtype=np.int32)
      neg_caption_masks = np.array(neg_caption_masks, dtype=np.int32)

      neg_captions = self.int2str(neg_captionids)
      deltas = trntst_util.get_scores(neg_captions, pos_vids, self.cider_scorer)

      yield {
        'fts': pos_fts,
        'captionids': pos_captionids,
        'caption_masks': pos_caption_masks,
        'neg_captionids': neg_captionids,
        'neg_caption_masks': neg_caption_masks,
        'deltas': deltas,
      }


class ValReader(framework.model.data.Reader):
  def __init__(self, ft_files, annotation_file, label_file):
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.gts = []

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
    self.num_caption = self.captionids.shape[0]

    with open(label_file) as f:
      vid2gid = {}
      for line in f:
        line = line.strip()
        data = line.split(' ')
        vid = int(data[0])
        gid = int(data[1])
        vid2gid[vid] = gid
    for vid in range(len(vid2gid)):
      self.gts.append(vid2gid[vid])

  def yield_val_batch(self, batch_size):
    cnt = 0
    for ft, gt in zip(self.fts, self.gts):
      fts = np.expand_dims(ft, 0)
      fts = np.repeat(fts, self.num_caption, 0)
      yield {
        'fts': fts,
        'captionids': self.captionids,
        'caption_masks': self.caption_masks,
        'gt': gt,
      }
      cnt += 1
      # if cnt % 10 == 0:
      #   print cnt
      if cnt == 200:
        break


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
    self.num_caption = self.captionids.shape[0]

  def yield_tst_batch(self, batch_size):
    for ft in self.fts:
      fts = np.expand_dims(ft, 0)
      fts = np.repeat(fts, self.num_caption, 0)
      yield {
        'fts': fts,
        'captionids': self.captionids,
        'caption_masks': self.caption_masks,
      }
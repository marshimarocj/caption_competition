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
    self.window_sizes = [3, 4, 5]
    self.num_filters = [100, 100, 100]
    self.pool = 'mean'

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = 0.1
    self.alpha = 0.5
    self.num_neg = 1
    self.l2norm = True


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
  cfg.num_neg = kwargs['num_neg']
  cfg.l2norm = kwargs['l2norm']
  cfg.dim_ft = kwargs['dim_ft']
  cfg.dim_joint_embed = kwargs['dim_joint_embed']

  cfg.max_words_in_caption = kwargs['max_words_in_caption']
  cfg.window_sizes = kwargs['window_sizes']
  cfg.num_filters = kwargs['num_filters']
  cfg.pool = kwargs['pool']

  we_cfg = cfg.subcfgs[WE]
  # we_cfg.dim_embed = kwargs['dim_embed']
  # we_cfg.num_words = kwargs['num_words']
  we_cfg.lr_mult = 1e-2

  # print we_cfg

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'ceve.Model'

  class InKey(enum.Enum):
    FT = 'ft'
    CAPTIONID = 'captionid'
    CAPTION_MASK = 'caption_mask'

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

    return {
      self.InKey.FT: fts,
      self.InKey.CAPTIONID: captionids,
      self.InKey.CAPTION_MASK: caption_masks,
    }

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      dim_word_embed = self._config.subcfgs[WE].dim_embed
      self.conv_Ws = []
      self.conv_Bs = []
      for window_size, num_filter in zip(self._config.window_sizes, self._config.num_filters):
        filter_shape = [window_size, dim_word_embed, num_filter]
        scale = 1.0 / (dim_word_embed*window_size)**0.5
        conv_W = tf.contrib.framework.model_variable('conv_%d_W'%window_size,
          shape=filter_shape, dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        conv_B = tf.contrib.framework.model_variable('conv_%d_B'%window_size,
          shape=(num_filter,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self.conv_Ws.append(conv_W)
        self.conv_Bs.append(conv_B)
        self._weights.append(conv_W)
        self._weights.append(conv_B)

      self.pca_W = tf.contrib.framework.model_variable('pca_W',
        shape=(self._config.dim_ft, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.pca_B = tf.contrib.framework.model_variable('pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.pca_W)
      self._weights.append(self.pca_B)

  def encode_caption(self, wvec, mask):
    print wvec.get_shape()
    batch_size = tf.shape(wvec)[0]
    mask = tf.to_float(tf.expand_dims(mask, 2))
    pools = []
    for i in range(len(self.config.window_sizes)):
      conv_W = self.conv_Ws[i]
      conv_B = self.conv_Bs[i]
      window_size = self._config.window_sizes[i]
      num_filter = self._config.num_filters[i]

      conv_out = tf.nn.conv1d(wvec, conv_W, 1, 'SAME', 
        name='conv_%d'%window_size) # use SAME padding for the ease of mask operation
      conv_out = tf.nn.bias_add(conv_out, conv_B)
      conv_out = tf.nn.tanh(conv_out)
      conv_out = conv_out * mask
      print conv_out.get_shape()
      if self._config.pool == 'mean':
        pool = tf.reduce_sum(conv_out, 1) / tf.reduce_sum(mask, 1)
      else:
        _mask = tf.cast(mask, tf.bool)
        _mask = tf.tile(_mask, [1, 1, num_filter])
        conv_out = tf.where(_mask, conv_out, -10*tf.ones_like(conv_out, dtype=tf.float32))
        pool = tf.reduce_max(conv_out, 1)
      pools.append(pool) # (None, num_filter)
    caption_embed = tf.concat(pools, 1)
    print caption_embed.get_shape()
    return caption_embed

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    encoder = self.submods[WE]
    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.CAPTION: in_ops[self.InKey.CAPTIONID],
      }, mode)
    wvecs = out_ops[encoder.OutKey.EMBED] # (None, max_words_in_caption, dim_embed)

    with tf.variable_scope(self.name_scope):
      caption_embed = self.encode_caption(wvecs, in_ops[self.InKey.CAPTION_MASK])
      if self._config.l2norm:
        caption_embed = tf.nn.l2_normalize(caption_embed, 1)

      ft_embed = tf.nn.xw_plus_b(in_ops[self.InKey.FT], self.pca_W, self.pca_B)
      ft_embed = tf.nn.tanh(ft_embed)
      if self._config.l2norm:
        ft_embed = tf.nn.l2_normalize(ft_embed, 1)

    def trn(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        pos_ft_embed = ft_embed[:-self._config.num_neg]
        pos_caption_embed = caption_embed[:-self._config.num_neg]
        neg_ft_embed = ft_embed[-self._config.num_neg:]
        neg_caption_embed = caption_embed[-self._config.num_neg:]

        pos_sim = tf.reduce_sum(pos_ft_embed * pos_caption_embed, 1) # (trn_batch_size,)
        neg_caption_sim = tf.matmul(pos_ft_embed, neg_caption_embed, transpose_b=True) # (trn_batch_size, neg)
        neg_ft_sim = tf.matmul(pos_caption_embed, neg_ft_embed, transpose_b=True) # (trn_batch_size, neg)

        neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1) / 100. # (trn_batch_size,)
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1) / 100. # (trn_batch_size,)

      return pos_sim, neg_caption_sim, neg_ft_sim

    def tst(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        sim = tf.matmul(ft_embed, caption_embed, transpose_b=True)
      return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      pos_sim, neg_caption_sim, neg_ft_sim = trn(ft_embed, caption_embed)
      sim = tst(ft_embed, caption_embed)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_caption_sim,
      }
    else:
      sim = tst(ft_embed, caption_embed)
      return {
        self.OutKey.SIM: sim,
      }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      pos_sim = self._outputs[self.OutKey.P_SIM]
      neg_caption_sim = self._outputs[self.OutKey.NC_SIM]
      neg_ft_sim = self._outputs[self.OutKey.NF_SIM]

      contrast_caption_loss = neg_caption_sim + self._config.margin - pos_sim
      contrast_caption_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_caption_loss))
      self.op2monitor['contrast_caption_loss'] = contrast_caption_loss

      contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
      contrast_ft_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_ft_loss))
      self.op2monitor['contrast_ft_loss'] = contrast_ft_loss

      loss = self._config.alpha * contrast_caption_loss + (1.0 - self._config.alpha) * contrast_ft_loss
      loss = tf.reduce_sum(loss)
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


class TrnReader(framework.model.data.Reader):
  def __init__(self, num_neg, ft_files, annotation_file):
    self.num_neg = num_neg
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.idxs = []
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

    self.num_caption = self.captionids.shape[0]
    self.idxs = range(self.num_caption)

  def num_record(self):
    return self.num_caption

  def reset(self):
    random.shuffle(self.idxs)

  def yield_trn_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      pos_idxs = self.idxs[i:i+batch]
      pos_ft_idxs = set(self.ft_idxs[pos_idxs].tolist())

      pos_fts = self.fts[pos_idxs]
      pos_captionids = self.captionids[pos_idxs]
      pos_caption_masks = self.caption_masks[pos_idxs]

      idxs = range(self.num_caption)
      random.shuffle(idxs)

      neg_fts = []
      neg_captionids= []
      neg_caption_masks = []
      for idx in idxs:
        if self.ft_idxs[i] not in pos_ft_idxs:
          neg_fts.append(self.fts[idx])
          neg_captionids.append(self.captionids[idx])
          neg_caption_masks.append(self.caption_masks[idx])
          if len(neg_fts) == self.num_neg:
            break
      neg_fts = np.array(neg_fts, dtype=np.float32)
      neg_captionids = np.array(neg_captionids, dtype=np.int32)
      neg_caption_masks = np.array(neg_caption_masks, dtype=np.int32)

      fts = np.concatenate([pos_fts, neg_fts], 0)
      captionids = np.concatenate([pos_captionids, neg_captionids], 0)
      caption_masks = np.concatenate([pos_caption_masks, neg_caption_masks], 0)

      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
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
    for ft, gt in zip(self.fts, self.gts):
      fts = np.expand_dims(ft, 0)
      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
        'gt': gt,
      }


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
    for ft in self.fts:
      fts = np.expand_dims(ft, 0)
      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
      }

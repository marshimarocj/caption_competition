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

WE = 'word'
RNN = 'rnn'
CELL = encoder.birnn.CELL
RCELL = encoder.birnn.RCELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[WE] = encoder.word.Config()
    self.subcfgs[RNN] = encoder.birnn.Config()

    self.max_words_in_caption = 30
    self.pool = 'mean'

    self.dim_ft = 1024
    self.dim_joint_embed = 512
    self.margin = 0.1
    self.alpha = 0.5
    self.beta = 0.5
    self.num_neg = 1
    self.l2norm = True
    self.num_concept = -1

  def _assert(self):
    assert self.max_words_in_caption == self.subcfgs[RNN].num_step
    assert self.subcfgs[WE].dim_embed == self.subcfgs[RNN].subcfgs[CELL].dim_input


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
  cfg.pool = kwargs['pool']

  cfg.num_concept = kwargs['num_concept']
  cfg.beta = kwargs['beta']

  we_cfg = cfg.subcfgs[WE]
  we_cfg.lr_mult = kwargs['lr_mult']

  rnn_cfg = cfg.subcfgs[RNN]
  rnn_cfg.num_step = kwargs['max_words_in_caption']
  rnn_cfg.cell_type = kwargs['cell']

  for cell in [CELL, RCELL]:
    cell_cfg = rnn_cfg.subcfgs[cell]
    cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
    cell_cfg.dim_input = 300
    cell_cfg.keepout_prob = 0.5
    cell_cfg.keepin_prob = 0.5

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'rnnve.Model'

  class InKey(enum.Enum):
    FT = 'ft'
    CAPTIONID = 'captionid'
    CAPTION_MASK = 'caption_mask'
    IS_TRN = 'is_trn'
    LABEL = 'label'

  class OutKey(enum.Enum):
    SIM = 'sim'
    P_SIM = 'pos_sim'
    NF_SIM = 'neg_ft_sim'
    NC_SIM = 'neg_caption_sim'
    LOGIT = 'logit'

  def _set_submods(self):
    return {
      WE: encoder.word.Encoder(self._config.subcfgs[WE]),
      RNN: encoder.birnn.Encoder(self._config.subcfgs[RNN]),
    }

  def _add_input_in_mode(self, mode):
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, self._config.dim_ft), name='fts')
      captionids = tf.placeholder(
        tf.int32, shape=(None, self._config.max_words_in_caption), name='captionids')
      caption_masks = tf.placeholder(
        tf.int32, shape=(None, self._config.max_words_in_caption), name='caption_masks')
      labels = tf.placeholder(
        tf.float32, shape=(None, self._config.num_concept), name='label')
      is_trn = tf.placeholder(tf.bool, shape=(), name='is_trn')

    return {
      self.InKey.FT: fts,
      self.InKey.CAPTIONID: captionids,
      self.InKey.CAPTION_MASK: caption_masks,
      self.InKey.LABEL: labels,
      self.InKey.IS_TRN: is_trn,
    }

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      dim_hidden = self._config.subcfgs[RNN].subcfgs[CELL].dim_hidden
      self.caption_pca_W = tf.contrib.framework.model_variable('caption_pca_W',
        shape=(2*dim_hidden, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.caption_pca_B = tf.contrib.framework.model_variable('caption_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.caption_pca_W)
      self._weights.append(self.caption_pca_B)

      self.ft_pca_W = tf.contrib.framework.model_variable('ft_pca_W',
        shape=(self._config.dim_ft, self._config.dim_joint_embed), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.ft_pca_B = tf.contrib.framework.model_variable('ft_pca_B',
        shape=(self._config.dim_joint_embed,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.ft_pca_W)
      self._weights.append(self.ft_pca_B)

      self.fc_W = tf.contrib.framework.model_variable('fc_W',
        shape=(self._config.dim_joint_embed, self._config.num_concept), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.fc_B = tf.contrib.framework.model_variable('fc_B',
        shape=(self._config.num_concept), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.fc_W)
      self._weights.append(self.fc_B)

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

    def trn(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        logit = tf.nn.xw_plus_b(ft_embed, self.fc_W, self.fc_B)

        pos_ft_embed = ft_embed[:-self._config.num_neg]
        pos_caption_embed = caption_embed[:-self._config.num_neg]
        neg_ft_embed = ft_embed[-self._config.num_neg:]
        neg_caption_embed = caption_embed[-self._config.num_neg:]

        pos_sim = tf.reduce_sum(pos_ft_embed * pos_caption_embed, 1) # (trn_batch_size,)
        neg_caption_sim = tf.matmul(pos_ft_embed, neg_caption_embed, transpose_b=True) # (trn_batch_size, neg)
        neg_ft_sim = tf.matmul(pos_caption_embed, neg_ft_embed, transpose_b=True) # (trn_batch_size, neg)

        neg_caption_sim = tf.reduce_logsumexp(100.*neg_caption_sim, 1) / 100. # (trn_batch_size,)
        neg_ft_sim = tf.reduce_logsumexp(100.*neg_ft_sim, 1) / 100. # (trn_batch_size,)

      return pos_sim, neg_caption_sim, neg_ft_sim, logit

    def tst(ft_embed, caption_embed):
      with tf.variable_scope(self.name_scope):
        sim = tf.matmul(ft_embed, caption_embed, transpose_b=True)
      return sim

    if mode == framework.model.module.Mode.TRN_VAL:
      pos_sim, neg_caption_sim, neg_ft_sim, logit = trn(ft_embed, caption_embed)
      sim = tst(ft_embed, caption_embed)
      return {
        self.OutKey.SIM: sim,
        self.OutKey.P_SIM: pos_sim,
        self.OutKey.NF_SIM: neg_ft_sim,
        self.OutKey.NC_SIM: neg_caption_sim,
        self.OutKey.LOGIT: logit,
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
      logit = self._outputs[self.OutKey.LOGIT]
      labels = self._inputs[self.InKey.LABEL]

      contrast_caption_loss = neg_caption_sim + self._config.margin - pos_sim
      contrast_caption_loss = tf.maximum(contrast_caption_loss, tf.zeros_like(contrast_caption_loss))
      self.op2monitor['contrast_caption_loss'] = tf.reduce_sum(contrast_caption_loss)

      contrast_ft_loss = neg_ft_sim + self._config.margin - pos_sim
      contrast_ft_loss = tf.maximum(contrast_ft_loss, tf.zeros_like(contrast_ft_loss))
      self.op2monitor['contrast_ft_loss'] = tf.reduce_sum(contrast_ft_loss)

      loss = self._config.alpha * contrast_caption_loss + (1.0 - self._config.alpha) * contrast_ft_loss
      concept_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logit), -1)
      self.op2monitor['concept_loss'] = tf.reduce_sum(concept_loss)
      loss += self._config.beta * concept_loss
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


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    super(PathCfg, self).__init__()
    # manually provided in the cfg file
    self.output_dir = ''
    self.trn_ftfiles = []
    self.val_ftfiles = []
    self.tst_ftfiles = []

    self.val_label_file = ''
    self.word_file = ''
    self.embed_file = ''

    self.trn_annotation_file = ''
    self.val_annotation_file = ''
    self.tst_annotation_file = ''

    self.concept_wid_file = ''

    # automatically generated paths
    self.log_file = ''


class TrnTst(trntst_util.TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
      self.model.inputs[self.model.InKey.LABEL]: data['labels'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }


class TrnReader(framework.model.data.Reader):
  def __init__(self, num_neg, num_concept, ft_files, annotation_file, concept_wid_file):
    self.num_neg = num_neg
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.labels = np.empty(0)
    self.idxs = []
    self.num_caption = 0

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(fts, axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    self.num_caption = self.captionids.shape[0]
    self.idxs = range(self.num_caption)

    with open(concept_wid_file) as f:
      wids = json.load(f)
    wid2lid = {}
    for i, wid in enumerate(wids):
      wid2lid[wid] = i
    self.labels = []
    for captionid in self.captionids:
      label = np.zeros((num_concept,), dtype=np.float32)
      for wid in captionid:
        if wid == 1:
          break
        if wid in wid2lid:
          lid = wid2lid[wid]
          label[lid] = 1.
      self.labels.append(label)
    self.labels = np.array(self.labels, dtype=np.float32)

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

      pos_labels = self.labels[pos_idxs]

      idxs = range(self.num_caption)
      random.shuffle(idxs)

      neg_fts = []
      neg_captionids= []
      neg_caption_masks = []
      neg_labels = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        if ft_idx not in pos_ft_idxs:
          neg_fts.append(self.fts[ft_idx])
          neg_captionids.append(self.captionids[idx])
          neg_caption_masks.append(self.caption_masks[idx])
          neg_labels.append(self.labels[idx])
          if len(neg_fts) == self.num_neg:
            break
      neg_fts = np.array(neg_fts, dtype=np.float32)
      neg_captionids = np.array(neg_captionids, dtype=np.int32)
      neg_caption_masks = np.array(neg_caption_masks, dtype=np.int32)
      neg_labels = np.array(neg_labels, dtype=np.float32)

      fts = np.concatenate([pos_fts, neg_fts], 0)
      captionids = np.concatenate([pos_captionids, neg_captionids], 0)
      caption_masks = np.concatenate([pos_caption_masks, neg_caption_masks], 0)
      labels = np.concatenate([pos_labels, neg_labels], 0)

      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
        'labels': labels,
      }


ValReader = trntst_util.ValReader
TstReader = trntst_util.TstReader

import enum
import sys
import os
import cPickle
import json
import random
sys.path.append('../')

import tensorflow as tf
import numpy as np

import framework.model.module
import framework.model.trntst
import framework.model.data
import framework.util.caption.utility
from framework.model.module import Mode

import framework.impl.encoder.dnn
import decoder.rnn
from decoder.rnn import CELL
import service.fast_cider
import trntst_util


VE = 'encoder'
VD = 'decoder'
CELL = decoder.rnn.CELL


class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[VE] = framework.impl.encoder.dnn.Config()
    self.subcfgs[VD] = decoder.rnn.Config()

    self.reward_alpha = .5
    self.num_sample = 5
    self.tst_strategy = 'beam' # greedy, sample, beam
    self.tst_num_sample = 100
    self.tst_sample_topk = -1
    self.reward_metric = 'cider'

    self.min_ngram_in_diversity = 0
    self.max_ngram_in_diversity = 4


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.monitor_iter = 100
  cfg.val_loss = False
  cfg.monitor_iter = 100
  cfg.trn_batch_size = 64
  cfg.tst_batch_size = 128
  cfg.base_lr = 1e-5
  cfg.num_epoch = kwargs['num_epoch']
  cfg.reward_alpha = kwargs['reward_alpha']
  cfg.reward_metric = kwargs['reward_metric']
  cfg.num_sample = kwargs['num_sample']
  cfg.tst_strategy = kwargs['tst_strategy']
  cfg.tst_num_sample = kwargs['tst_num_sample']
  cfg.tst_sample_topk = kwargs['tst_sample_topk']
  cfg.min_ngram_in_diversity = kwargs['min_ngram_in_diversity']
  cfg.max_ngram_in_diversity = kwargs['max_ngram_in_diversity']

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


class Model(framework.model.module.AbstractPGModel):
  name_scope = 'vevd.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    ROLL_CAPTIONID = 'rollout_captionids'
    ROLL_CAPTIONMASK = 'rollout_caption_masks'
    IS_TRN = 'is_training'
    REWARD = 'reward'

  class OutKey(enum.Enum):
    ROLL_LOGPROB = 'roll_log_prob'

    OUT_WID = 'out_wid'
    BEAM_CUM_LOG_PROB = 'beam_cum_log_prob'
    BEAM_PRE = 'beam_pre'
    BEAM_END = 'beam_end'
    BASELINE_OUT_WID = 'greedy_out_wid'
    ROLL_OUT_WID = 'roll_out_wid'

    SAMPLE_OUT_WID = 'sample_out_wid'
    LOG_PROB = 'sample_log_prob'

  def _set_submods(self):
    return {
      VE: framework.impl.encoder.dnn.Encoder(self._config.subcfgs[VE]),
      VD: decoder.rnn.Decoder(self._config.subcfgs[VD]),
    }

  def _add_input_in_mode(self, mode):
    dim_fts = self._config.subcfgs[VE].dim_fts
    num_step = self._config.subcfgs[VD].num_step

    if mode == framework.model.module.Mode.TRN_VAL:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(dim_fts)), name=self.InKey.FT.value)
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)
        # trn only
        roll_captionids = tf.placeholder(
          tf.int32, shape=(None, self._config.num_sample, num_step), name=self.InKey.ROLL_CAPTIONID.value)
        roll_caption_masks = tf.placeholder(
          tf.float32, shape=(None, self._config.num_sample, num_step), name=self.InKey.ROLL_CAPTIONMASK.value)
        reward = tf.placeholder(
          tf.float32, shape=(None, self._config.num_sample), name=self.InKey.REWARD.value)

        return {
          self.InKey.FT: fts,
          self.InKey.ROLL_CAPTIONID: roll_captionids,
          self.InKey.ROLL_CAPTIONMASK: roll_caption_masks,
          self.InKey.IS_TRN: is_training,
          self.InKey.REWARD: reward,
        }
    else:
      with tf.variable_scope(self.name_scope):
        fts = tf.placeholder(
          tf.float32, shape=(None, sum(dim_fts)), name='fts')
        is_training = tf.placeholder(
          tf.bool, shape=(), name=self.InKey.IS_TRN.value)

        return {
          self.InKey.FT: fts,
          self.InKey.IS_TRN: is_training,
        }

  def _build_parameter_graph(self):
    pass

  def get_out_ops_in_mode(self, in_ops, mode):
    num_sample = self._config.num_sample
    num_step = self._config.subcfgs[VD].num_step
    dim_output = self._config.subcfgs[VE].dim_output

    encoder = self.submods[VE]
    decoder = self.submods[VD]

    out_ops = encoder.get_out_ops_in_mode({
      encoder.InKey.FT: in_ops[self.InKey.FT],
      encoder.InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
    }, mode)
    ft_embed = out_ops[encoder.OutKey.EMBED] # (None, dim_output)
    with tf.variable_scope(self.name_scope):
      batch_size = tf.shape(ft_embed)[0]
      init_wid = tf.zeros((batch_size,), dtype=tf.int32)

    def trn_val(ft_embed, init_wid):
      # val
      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, Mode.VAL,
        search_strategy='greedy')
      greedy_out_wid = out_ops[decoder.OutKey.OUT_WID]

      # reinforce ops
      with tf.variable_scope(self.name_scope):
        expanded_init_wid = tf.zeros((batch_size*num_sample,), dtype=tf.int32) # (None*num_sample,)
        expanded_ft_embed = tf.tile(tf.expand_dims(ft_embed, 1), [1, num_sample, 1])
        expanded_ft_embed = tf.reshape(expanded_ft_embed, [-1, dim_output]) # (None*num_sample, dim_output)
        roll_captionids = in_ops[self.InKey.ROLL_CAPTIONID]
        roll_captionids = tf.reshape(roll_captionids, (-1, num_step)) # (None*num_sample, num_step)

      vd_inputs = {
        decoder.InKey.FT: expanded_ft_embed,
        decoder.InKey.INIT_WID: expanded_init_wid,
        decoder.InKey.CAPTIONID: roll_captionids,
      }
      out_ops = decoder.get_out_ops_in_mode(vd_inputs, Mode.TRN, is_trn=False)
      roll_log_prob = out_ops[decoder.OutKey.LOG_PROB]
      with tf.variable_scope(self.name_scope):
        roll_log_prob = tf.reshape(roll_log_prob, [-1, num_sample, num_step-1])

      return {
        self.OutKey.ROLL_LOGPROB: roll_log_prob, # (None, num_sample, num_step-1)
        self.OutKey.OUT_WID: greedy_out_wid,
      }

    def rollout(ft_embed, init_wid):
      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
      }

      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode,
        search_strategy='greedy')
      baseline_out_wid = out_ops[decoder.OutKey.OUT_WID]

      out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
        search_strategy='sample', topk=-1, num_sample=num_sample)
      roll_out_wid = out_ops[decoder.OutKey.OUT_WID]

      return {
        self.OutKey.BASELINE_OUT_WID: baseline_out_wid,
        self.OutKey.ROLL_OUT_WID: roll_out_wid,
      }

    def tst(ft_embed, init_wid):
      vd_inputs = {
        decoder.InKey.FT: ft_embed,
        decoder.InKey.INIT_WID: init_wid,
      }
      if self._config.tst_strategy == 'greedy':
        out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
          search_strategy='greedy', task='generation')
        return {
          self.OutKey.OUT_WID: out_ops[decoder.OutKey.OUT_WID],
        }
      elif self._config.tst_strategy == 'beam':
        out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
          search_strategy='beam', task='generation')
        return {
          self.OutKey.OUT_WID: out_ops[decoder.OutKey.OUT_WID],
          self.OutKey.BEAM_CUM_LOG_PROB: out_ops[decoder.OutKey.BEAM_CUM_LOG_PROB],
          self.OutKey.BEAM_PRE: out_ops[decoder.OutKey.BEAM_PRE],
          self.OutKey.BEAM_END: out_ops[decoder.OutKey.BEAM_END],
        }
      elif self._config.tst_strategy == 'sample':
        out_ops = decoder.get_out_ops_in_mode(vd_inputs, mode, 
          search_strategy='sample', num_sample=self._config.tst_num_sample, topk=self._config.tst_sample_topk, task='generation')
        return {
          self.OutKey.OUT_WID: out_ops[decoder.OutKey.OUT_WID],
          self.OutKey.LOG_PROB: out_ops[decoder.OutKey.LOG_PROB],
        }

    delegate = {
      Mode.TRN_VAL: trn_val,
      Mode.ROLLOUT: rollout,
      Mode.TST: tst,
    }
    return delegate[mode](ft_embed, init_wid)

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      reward = self._inputs[self.InKey.REWARD]
      roll_mask = self._inputs[self.InKey.ROLL_CAPTIONMASK]
      roll_mask = roll_mask[:, :, 1:]
      roll_log_probs = self._outputs[self.OutKey.ROLL_LOGPROB] # (None, num_sample, max_words_in_caption)
      surrogate_loss = -tf.expand_dims(reward, axis=2) * roll_log_probs * roll_mask
      surrogate_loss = tf.reduce_sum(surrogate_loss) / tf.reduce_sum(roll_mask)
      self.op2monitor['surrogate_loss'] = surrogate_loss
      self.op2monitor['reward'] = tf.reduce_mean(reward)

    return surrogate_loss

  def op_in_rollout(self, **kwargs):
    op_dict = {
      self.OutKey.BASELINE_OUT_WID: self._rollout_outputs[self.OutKey.BASELINE_OUT_WID],
      self.OutKey.ROLL_OUT_WID: self._rollout_outputs[self.OutKey.ROLL_OUT_WID],
    }
    return op_dict

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
    }

  def op_in_tst(self, **kwargs):
    if self._config.tst_strategy == 'beam':
      return {
        self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
        self.OutKey.BEAM_CUM_LOG_PROB: self._outputs[self.OutKey.BEAM_CUM_LOG_PROB],
        self.OutKey.BEAM_PRE: self._outputs[self.OutKey.BEAM_PRE],
        self.OutKey.BEAM_END: self._outputs[self.OutKey.BEAM_END],
      }
      return op_dict
    elif self._config.tst_strategy == 'greedy':
      return {
        self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
      }
    elif self._config.tst_strategy == 'sample':
      return {
        self.OutKey.OUT_WID: self._outputs[self.OutKey.OUT_WID],
        self.OutKey.LOG_PROB: self._outputs[self.OutKey.LOG_PROB],
      }


PathCfg = trntst_util.PathCfg


class TrnTst(framework.model.trntst.PGTrnTst):
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    framework.model.trntst.TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = framework.util.caption.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

    gt_file = path_cfg.groundtruth_file
    with open(gt_file) as f:
      self.vid2captions = cPickle.load(f)
    self.cider_scorer = service.fast_cider.CiderScorer()
    self.cider_scorer.init_refs(self.vid2captions)

  def feed_data_and_rollout(self, data, sess):
    op_dict = self.model.op_in_rollout()

    feed_dict = self._construct_feed_dict_in_rollout(data)
    baseline_out_wids, roll_out_wids = sess.run(
      [
        op_dict[self.model.OutKey.BASELINE_OUT_WID], 
        op_dict[self.model.OutKey.ROLL_OUT_WID], 
      ],
      feed_dict=feed_dict)

    vids = data['vids']
    if self.model_cfg.reward_metric == 'cider':
      quality_baselines = trntst_util.eval_cider_in_rollout(baseline_out_wids, vids, self.int2str, self.cider_scorer) # (None, 1)
      quality_rewards = trntst_util.eval_cider_in_rollout(roll_out_wids, vids, self.int2str, self.cider_scorer) # (None, num_sample)
      quality_rewards -= quality_baselines # (None, num_sample)
    elif self.model_cfg.reward_metric == 'bcmr':
      quality_baselines = trntst_util.eval_BCMR_in_rollout(baseline_out_wids, vids, self.int2str, self.cider_scorer, self.vid2captions)
      quality_rewards = trntst_util.eval_BCMR_in_rollout(roll_out_wids, vids, self.int2str, self.cider_scorer, self.vid2captions)
      quality_rewards -= quality_baselines
      quality_rewards /= 3.

    diverse_rewards = trntst_util.eval_bleu_diversity_in_rollout(roll_out_wids, self.int2str, 
      min_ngram=self.model_cfg.min_ngram_in_diversity, max_ngram=self.model_cfg.max_ngram_in_diversity)
    diverse_baselines = np.mean(diverse_rewards, axis=1)
    diverse_baselines = np.maximum(diverse_baselines, -.5 * np.ones(diverse_baselines.shape))
    diverse_rewards -= np.expand_dims(diverse_baselines, 1) # (None, num_sample)

    roll_caption_ids, roll_caption_masks = trntst_util.gen_captionid_masks_from_wids(roll_out_wids)

    data['rewards'] = (1 - self.model_cfg.reward_alpha) * quality_rewards + self.model_cfg.reward_alpha * diverse_rewards # (None, num_sample)
    data['roll_caption_ids'] = roll_caption_ids # (None, num_sample, num_step)
    data['roll_caption_masks'] = roll_caption_masks # (None, num_sample, num_step)
    return data

  def _construct_feed_dict_in_rollout(self, data):
    fts = data['fts']
    batch_size = fts.shape[0]

    return {
      self.model.rollout_inputs[self.model.InKey.FT]: fts,
      self.model.rollout_inputs[self.model.InKey.IS_TRN]: False,
    }

  def _construct_feed_dict_in_trn(self, data):
    fts = data['fts']
    captionids = data['captionids']
    caption_masks = data['caption_masks']
    roll_caption_ids = data['roll_caption_ids']
    roll_caption_masks = data['roll_caption_masks']
    rewards = data['rewards']

    batch_size = fts.shape[0]
    return {
      self.model.inputs[self.model.InKey.FT]: fts,
      self.model.inputs[self.model.InKey.IS_TRN]: True,
      self.model.inputs[self.model.InKey.ROLL_CAPTIONID]: roll_caption_ids,
      self.model.inputs[self.model.InKey.ROLL_CAPTIONMASK]: roll_caption_masks,
      self.model.inputs[self.model.InKey.REWARD]: rewards,
    }

  def _construct_feed_dict_in_val(self, data):
    fts = data['fts']
    captionids = data['captionids']
    caption_masks = data['caption_masks']

    return {
      self.model.inputs[self.model.InKey.FT]: fts,
      self.model.inputs[self.model.InKey.IS_TRN]: False,
    }

  def _construct_feed_dict_in_tst(self, data):
    fts = data['fts']

    return {
      self.model.inputs[self.model.InKey.FT]: fts,
      self.model.inputs[self.model.InKey.IS_TRN]: False,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    trntst_util.predict_and_eval_in_val(self, sess, tst_reader, metrics)

  def predict_in_tst(self, sess, tst_reader, predict_file):
    trntst_util.predict_in_tst(self, sess, tst_reader, predict_file, self.model_cfg.tst_strategy)


Reader = trntst_util.Reader

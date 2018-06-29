import enum
import math

import tensorflow as tf
from tensorflow.python.util import nest

import framework.model.module
import framework.util.expanded_op
import framework.impl.cell as cell

CELL = 'cell'


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.subcfgs[CELL] = cell.CellConfig()

    self.dim_input = 300
    self.dim_hidden = 300

    self.num_step = 30
    self.num_words = 10870

    self.beam_width = 5
    self.sent_pool_size = 5

  def _assert(self):
    assert self.dim_input == self.subcfgs[CELL].dim_input
    assert self.dim_hidden == self.subcfgs[CELL].dim_hidden


class Decoder(framework.model.module.AbstractModule):
  name_scope = 'vanilla.Decoder'

  class InKey(enum.Enum):
    FT = 'ft'
    CAPTIONID = 'captionids'
    INIT_WID = 'init_wid'

  class OutKey(enum.Enum):
    LOGIT = 'logit'
    LOG_PROB = 'log_prob'
    OUT_WID = 'out_wid'
    BEAM_CUM_LOG_PROB = 'beam_cum_log_prob'
    BEAM_PRE = 'beam_pre'
    BEAM_END = 'beam_end'

  def _set_submods(self):
    return {
      CELL: cell.LSTMCell(self._config.subcfgs[CELL]),
    }

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      scale = 1.0 / (self._config.num_words**0.5)
      self.word_embedding_W = tf.contrib.framework.model_variable('word_embedding_W',
        shape=(self._config.num_words, self._config.dim_input), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self._weights.append(self.word_embedding_W)

      scale = 1.0/ (self._config.dim_hidden**0.5)
      self.softmax_W = tf.contrib.framework.model_variable('softmax_W',
        shape=(self._config.dim_hidden, self._config.num_words), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self._weights.append(self.softmax_W)

      self.softmax_B = tf.contrib.framework.model_variable('softmax_B',
          shape=(self._config.num_words,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.softmax_B)

  def _ft_step(self, ft):
    state_size = self.submods[CELL].state_size
    states = [ft for _ in nest.flatten(state_size)]
    states = nest.pack_sequence_as(state_size, states)

    return states

  def _word_steps(self, captionids, state, is_trn):
    outputs = []
    log_probs = []
    batch_size = tf.shape(captionids)[0]
    row_idxs = tf.range(batch_size) # (None,)
    for i in xrange(self._config.num_step-1):
      input = tf.nn.embedding_lookup(self.word_embedding_W, captionids[:, i])
      out = self.submods[CELL].get_out_ops_in_mode({
        self.submods[CELL].InKey.INPUT: input,
        self.submods[CELL].InKey.STATE: state,
        self.submods[CELL].InKey.IS_TRN: is_trn,
        }, framework.model.module.Mode.TRN_VAL)
      output = out[self.submods[CELL].OutKey.OUTPUT] # (None, dim_hidden)
      state = out[self.submods[CELL].OutKey.STATE]

      logits = tf.nn.xw_plus_b(output, self.softmax_W, self.softmax_B)
      log_prob = tf.nn.log_softmax(logits)
      idxs = tf.stack([row_idxs, captionids[:, i+1]], axis=1)
      log_prob = tf.gather_nd(log_prob, idxs) # (None,)

      outputs.append(output)
      log_probs.append(log_prob)
    log_probs = tf.stack(log_probs, axis=1) # (None, num_step-1)

    outputs = tf.concat(outputs, 0) # ((num_step-1)*None, dim_hidden)
    logit_ops = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B) # ((num_step-1)*None, num_words)

    return logit_ops, log_probs

  def _greedy_word_steps(self, wordids, states):
    out_wids = []
    for i in xrange(self._config.num_step):
      input = tf.nn.embedding_lookup(self.word_embedding_W, wordids)
      out = self.submods[CELL].get_out_ops_in_mode({
        self.submods[CELL].InKey.INPUT: input,
        self.submods[CELL].InKey.STATE: states,
        self.submods[CELL].InKey.IS_TRN: False,
        }, framework.model.module.Mode.TST)
      outputs = out[self.submods[CELL].OutKey.OUTPUT]
      states = out[self.submods[CELL].OutKey.STATE]
      logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B) # (batch_size, num_words)
      wordids = tf.argmax(logits, axis=1)

      out_wids.append(wordids)
    out_wids = tf.stack(out_wids, axis=1) # (None, num_step)

    return out_wids

  def _beam_search_word_steps(self, wordids, state):
    next_step_func = next_step_func_handle(self, self.submods[CELL])
    op_groups = framework.util.expanded_op.beam_decode(
      next_step_func,
      wordids, state,
      self.submods[CELL].state_size, self._config.beam_width, self._config.num_step)
    return {
      self.OutKey.OUT_WID: op_groups[0],
      self.OutKey.BEAM_PRE: op_groups[1],
      self.OutKey.BEAM_CUM_LOG_PROB: op_groups[2],
      self.OutKey.BEAM_END: op_groups[3],
    }

  def _sample_topk_word_steps(self, wordids, states, num_sample, topk):
    out_wids = []
    state_struct = self.submods[CELL].state_size
    state_sizes = nest.flatten(state_struct)
    for i in xrange(self._config.num_step):
      input = tf.nn.embedding_lookup(self.word_embedding_W, wordids)
      out = self.submods[CELL].get_out_ops_in_mode({
        self.submods[CELL].InKey.INPUT: input,
        self.submods[CELL].InKey.STATE: states,
        self.submods[CELL].InKey.IS_TRN: False,
        }, framework.model.module.Mode.TST)
      outputs = out[self.submods[CELL].OutKey.OUTPUT]
      states = out[self.submods[CELL].OutKey.STATE]
      logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B) # (None, num_class)
      if i == 0:
        if topk == -1:
          wordids = tf.multinomial(logits, num_sample) # (None, num_sample)
          wordids = tf.reshape(wordids, (-1,)) # (None*num_sample)
        else:
          topk_logits, topk_idxs = tf.nn.top_k(logits, topk) # (None, topk)
          col_idxs = tf.cast(tf.multinomial(topk_logits, num_sample), tf.int32) # (None, num_sample)
          col_idxs = tf.reshape(col_idxs, (-1,))
          row_idxs = tf.range(tf.shape(topk_idxs)[0], dtype=tf.int32)
          row_idxs = tf.reshape(tf.tile(tf.expand_dims(row_idxs, 1), [1, num_sample]), (-1,))# [0, ...0, ..., None-1, ...None-1]
          idxs = tf.stack([row_idxs, col_idxs], axis=1) # (None*num_sample, 2)
          wordids = tf.gather_nd(topk_idxs, idxs) # (None*num_sample,)

        states = nest.flatten(states) # (batch_size, hidden_size)
        states = [
          tf.reshape(tf.tile(state, [1, num_sample]), (-1, state_size)) # (batch_size*num_sample, hidden_size)
          for state, state_size in zip(states, state_sizes)
        ]
        states = nest.pack_sequence_as(state_struct, states)
      else:
        if topk == -1:
          wordids = tf.multinomial(logits, 1)
          wordids = tf.reshape(wordids, (-1,)) # (None*num_sample,)
        else:
          topk_logits, topk_idxs = tf.nn.top_k(logits, topk) # (None*num_sample, topk)
          col_idxs = tf.cast(tf.multinomial(topk_logits, 1), tf.int32) # (None*num_sample, 1)
          col_idxs = tf.reshape(col_idxs, (-1,)) # (None*num_sample,)
          row_idxs = tf.range(tf.shape(topk_idxs)[0], dtype=tf.int32) # (None*num_sample,)
          idxs = tf.stack([row_idxs, col_idxs], axis=1) # (None*num_sample, 2)
          wordids = tf.gather_nd(topk_idxs, idxs) # (None*num_sample,)
      out_wids.append(wordids)
    out_wids = tf.stack(out_wids, axis=1) # (None*num_sample, num_step)
    out_wids = tf.reshape(out_wids, (-1, num_sample, self._config.num_step))
    out_wids = tf.cast(out_wids, tf.int32)

    return out_wids

  def get_out_ops_in_mode(self, in_ops, mode, reuse=True, **kwargs):
    def sample_ops():
      with tf.variable_scope(self.name_scope):
        fts = in_ops[self.InKey.FT]
        state = self._ft_step(fts)
        out_wids = self._sample_topk_word_steps(in_ops[self.InKey.INIT_WID], state, 
          kwargs['num_sample'], kwargs['topk']) # (None, num_sample, num_step)

        fts = tf.tile(tf.expand_dims(fts, 1), [1, kwargs['num_sample'], 1])
        fts = tf.reshape(fts, [-1, self._config.dim_hidden])
        state = self._ft_step(fts)
        out_wids = tf.reshape(out_wids, [-1, self._config.num_step])
        num = tf.shape(out_wids)[0]
        out_wids = tf.concat(
          [tf.zeros((num, 1), dtype=tf.int32), out_wids[:, :-1]], axis=1)
        logit_ops, log_probs = self._word_steps(out_wids, state, False)
        log_probs = tf.reshape(log_probs, (-1, kwargs['num_sample'], self._config.num_step-1))
        out_wids = tf.reshape(out_wids[:, 1:], (-1, kwargs['num_sample'], self._config.num_step-1))
      return out_wids, log_probs # (None, num_sample, num_step-1)

    def trn_val():
      assert 'is_trn' in kwargs

      with tf.variable_scope(self.name_scope):
        state = self._ft_step(in_ops[self.InKey.FT])
        logits, log_probs = self._word_steps(in_ops[self.InKey.CAPTIONID], state, kwargs['is_trn'])
        out_wids = self._greedy_word_steps(in_ops[self.InKey.INIT_WID], state)
      return {
        self.OutKey.LOGIT: logits,
        self.OutKey.LOG_PROB: log_probs,
        self.OutKey.OUT_WID: out_wids,
      }

    def trn():
      assert 'is_trn' in kwargs

      with tf.variable_scope(self.name_scope):
        state = self._ft_step(in_ops[self.InKey.FT])
        logits, log_probs = self._word_steps(in_ops[self.InKey.CAPTIONID], state, kwargs['is_trn'])
      return {
        self.OutKey.LOGIT: logits,
        self.OutKey.LOG_PROB: log_probs,
      }

    def val():
      assert 'search_strategy' in kwargs

      with tf.variable_scope(self.name_scope):
        if kwargs['search_strategy'] == 'sample':
          assert ('num_sample' in kwargs) and ('topk' in kwargs)

          out_wid, log_probs = sample_ops()
          return {
            self.OutKey.OUT_WID: out_wid,
            self.OutKey.LOG_PROB: log_probs,
          }
        elif kwargs['search_strategy'] == 'greedy':
          state = self._ft_step(in_ops[self.InKey.FT])
          out_wids = self._greedy_word_steps(in_ops[self.InKey.INIT_WID], state)
          return {
            self.OutKey.OUT_WID: out_wids,
          }

    def rollout():
      assert 'search_strategy' in kwargs

      with tf.variable_scope(self.name_scope):
        if kwargs['search_strategy'] == 'sample':
          assert ('num_sample' in kwargs) and ('topk' in kwargs)

          out_wids, log_probs = sample_ops()
          # print out_wids.get_shape()
        elif kwargs['search_strategy'] == 'greedy':
          state = self._ft_step(in_ops[self.InKey.FT])
          out_wids = self._greedy_word_steps(in_ops[self.InKey.INIT_WID], state)
          out_wids = tf.expand_dims(out_wids, 1) # (None, 1, num_step)
        return {
          self.OutKey.OUT_WID: out_wids,
        }

    def tst_generation():
      assert 'strategy' in kwargs

      if kwargs['strategy'] == 'beam':
        with tf.variable_scope(self.name_scope):
          state = self._ft_step(in_ops[self.InKey.FT])
          return self._beam_search_word_steps(in_ops[self.InKey.INIT_WID], state)
      elif kwargs['strategy'] == 'sample':
        assert ('topk' in kwargs) and ('num_sample' in kwargs)

        out_wids, log_probs = sample_ops()
        return {
          self.OutKey.OUT_WID: out_wids, # (None, num_sample, num_step-1)
          self.OutKey.LOG_PROB: log_probs, # (None, num_sample, num_step-1)
        }

    def tst_retrieval():
      with tf.variable_scope(self.name_scope):
        state = self._ft_step(in_ops[self.InKey.FT])
        logits, log_probs = self._word_steps(in_ops[self.InKey.CAPTIONID], state, False)
      return {
        self.OutKey.LOG_PROB: log_probs
      }

    def tst():
      assert 'task' in kwargs

      if kwargs['task'] == 'generation':
        return tst_generation()
      elif kwargs['task'] == 'retrieval':
        return tst_retrieval()

    delegate = {
      framework.model.module.Mode.TRN_VAL: trn_val,
      framework.model.module.Mode.TRN: trn,
      framework.model.module.Mode.VAL: val,
      framework.model.module.Mode.ROLLOUT: rollout,
      framework.model.module.Mode.TST: tst,
    }
    return delegate[mode]()


def next_step_func_handle(model, cell):
  def next_step_func(wordids, states, outputs, step):
    input = tf.nn.embedding_lookup(model.word_embedding_W, wordids) 
    out = cell.get_out_ops_in_mode({
      cell.InKey.INPUT: input,
      cell.InKey.STATE: states,
      cell.InKey.IS_TRN: False}, 
      framework.model.module.Mode.TST)
    outputs = out[cell.OutKey.OUTPUT]
    states = out[cell.OutKey.STATE]
    logit = tf.nn.xw_plus_b(outputs, model.softmax_W, model.softmax_B)
    log_prob = tf.nn.log_softmax(logit)
    return log_prob, states, None

  return next_step_func
 
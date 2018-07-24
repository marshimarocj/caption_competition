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

    self.num_ft = 11
    self.dim_ft = -1
    self.dim_input = 300
    self.dim_hidden = 300
    self.dim_attention = 512

    self.num_step = 30
    self.num_words = 10870

    self.beam_width = 5
    self.sent_pool_size = 5

  def _assert(self):
    assert self.dim_input + self.dim_ft == self.subcfgs[CELL].dim_input
    assert self.dim_hidden == self.subcfgs[CELL].dim_hidden


class Decoder(framework.model.module.AbstractModule):
  name_scope = 'attention.Decoder'

  class InKey(enum.Enum):
    INIT_FT = 'init_ft'
    FT = 'ft'
    FT_MASK = 'ft_mask'
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
      self.word_embedding_W = tf.get_variable('word_embedding_W',
        shape=(self._config.num_words, self._config.dim_input), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self._weights.append(self.word_embedding_W)

      scale = 1.0/ (self._config.dim_hidden**0.5)
      self.softmax_W = tf.get_variable('softmax_W',
        shape=(self._config.dim_hidden, self._config.num_words), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self._weights.append(self.softmax_W)

      self.softmax_B = tf.get_variable('softmax_B',
          shape=(self._config.num_words,), dtype=tf.float32,
          initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.softmax_B)

      rows = self._config.dim_hidden
      cols = self._config.dim_attention
      scale = 1.0 / (rows**0.5)
      self.W_a = tf.get_variable('W_a',
        shape=(rows, cols), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self._weights.append(self.W_a)
      rows = self._config.dim_ft
      scale = 1.0 / (rows**0.5)
      self.U_a = tf.get_variable('U_a',
        shape=(rows, cols), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self._weights.append(self.U_a)
      self.b_a = tf.get_variable('b_a',
        shape=(1, self._config.dim_attention), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.b_a)
      scale = 1.0 / (self._config.dim_attention**0.5)
      self.w_a = tf.get_variable('w_a',
        shape=(self._config.dim_attention, 1), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-scale, scale))
      self._weights.append(self.w_a)

  def _ft_step(self, init_ft):
    state_size = self.submods[CELL].state_size
    states = [init_ft for _ in nest.flatten(state_size)]
    states = nest.pack_sequence_as(state_size, states)

    return states

  def _gen_Uav_plus_ba_op(self, fts):
    fts = tf.expand_dims(fts, 2) # (None, num_ft, 1, dim_ft)
    U_a = tf.reshape(self.U_a, [1, 1, self._config.dim_ft, self._config.dim_attention])
    Uav = tf.nn.conv2d(fts, U_a, [1, 1, 1, 1], 'VALID') # (None, num_ft, 1, dim_attention)
    Uav = tf.squeeze(Uav) # (None, num_ft, dim_attention)

    Uav_plus_ba = Uav + self.b_a
    return Uav_plus_ba

  def _word_steps(self, init_ft, fts, ft_masks, captionids, state):
    outputs = []
    log_probs = []
    batch_size = tf.shape(captionids)[0]
    row_idxs = tf.range(batch_size) # (None,)
    Uav_plus_ba = self._gen_Uav_plus_ba_op(fts)
    output = init_ft
    for i in xrange(self._config.num_step-1):
      input = tf.nn.embedding_lookup(self.word_embedding_W, captionids[:, i])
     
      output, state, alpha = self._attention_on_recurrent(
        input, state, 
        Uav_plus_ba, output, fts, ft_masks, 
        True, framework.model.module.Mode.TRN_VAL)

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

  def _greedy_word_steps(self, init_ft, fts, ft_masks, wordids, states):
    out_wids = []
    Uav_plus_ba = self._gen_Uav_plus_ba_op(fts)
    outputs = init_ft
    for i in xrange(self._config.num_step):
      input = tf.nn.embedding_lookup(self.word_embedding_W, wordids)

      outputs, states, alphas = self._attention_on_recurrent(
        input, states, 
        Uav_plus_ba, outputs, fts, ft_masks,
        False, framework.model.module.Mode.TRN_VAL)

      logits = tf.nn.xw_plus_b(outputs, self.softmax_W, self.softmax_B) # (batch_size, num_words)
      wordids = tf.argmax(logits, axis=1)

      out_wids.append(wordids)
    out_wids = tf.stack(out_wids, axis=1) # (None, num_step)

    return out_wids

  def _gen_tiled_version(self, fts, ft_masks):
    shape = (-1, self._config.num_ft, self._config.dim_ft)
    tiled_fts = tf.tile(fts, [1, self._config.beam_width, 1])
    tiled_fts = tf.reshape(tiled_fts, shape)

    shape = (-1, self._config.num_ft)
    tiled_ft_masks = tf.tile(ft_masks, [1, self._config.beam_width])
    tiled_ft_masks = tf.reshape(tiled_ft_masks, shape)

    fts = tf.expand_dims(tiled_fts, 2) # (None, num_ft, 1, dim_ft)
    U_a = tf.reshape(self.U_a, [1, 1, self._config.dim_ft, self._config.dim_attention])
    Uav = tf.nn.conv2d(fts, U_a, [1, 1, 1, 1], 'VALID') # (None, num_ft, 1, dim_attention)
    Uav = tf.squeeze(Uav) # (None, num_ft, dim_attention)

    tiled_Uav_plus_ba = Uav + self.b_a

    return tiled_Uav_plus_ba, tiled_fts, tiled_ft_masks

  def _beam_search_word_steps(self, init_ft, fts, ft_masks, wordids, state):
    Uav_plus_ba = self._gen_Uav_plus_ba_op(fts)
    tiled_Uav_plus_ba, tiled_fts, tiled_ft_masks = self._gen_tiled_version(fts, ft_masks)

    next_step_func = next_step_func_handle(self, 
      [Uav_plus_ba, tiled_Uav_plus_ba], [fts, tiled_fts], [ft_masks, tiled_ft_masks])

    op_groups = framework.util.expanded_op.beam_decode(
      next_step_func,
      wordids, state,
      self.submods[CELL].state_size, self._config.beam_width, self._config.num_step,
      init_output=init_ft)
    return {
      self.OutKey.OUT_WID: op_groups[0],
      self.OutKey.BEAM_PRE: op_groups[1],
      self.OutKey.BEAM_CUM_LOG_PROB: op_groups[2],
      self.OutKey.BEAM_END: op_groups[3],
    }

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      if mode == framework.model.module.Mode.TRN_VAL:
        state = self._ft_step(in_ops[self.InKey.INIT_FT])
        logits, log_probs = self._word_steps(
          in_ops[self.InKey.INIT_FT], in_ops[self.InKey.FT], in_ops[self.InKey.FT_MASK], in_ops[self.InKey.CAPTIONID], state)

        out_wids = self._greedy_word_steps(
          in_ops[self.InKey.INIT_FT], in_ops[self.InKey.FT], in_ops[self.InKey.FT_MASK], in_ops[self.InKey.INIT_WID], state)

        out = {
          self.OutKey.LOGIT: logits,
          self.OutKey.LOG_PROB: log_probs,
          self.OutKey.OUT_WID: out_wids,
        }
        return out
      else:
        state = self._ft_step(in_ops[self.InKey.INIT_FT])
        return self._beam_search_word_steps(
          in_ops[self.InKey.INIT_FT], in_ops[self.InKey.FT], in_ops[self.InKey.FT_MASK], in_ops[self.InKey.INIT_WID], state)

  def _attention_on_recurrent(self, x, states, 
      Uav_plus_ba, h, fts, ft_masks, 
      is_trn, mode):
    hW = tf.matmul(h, self.W_a) # (None, dim_attention)
    hW = tf.expand_dims(hW, 1) # (None, 1, dim_attention)
    w_a = tf.reshape(self.w_a, (1, 1, -1))
    e = tf.reduce_sum(w_a * tf.tanh(hW + Uav_plus_ba), 2) # (None, num_ft)
    alphas = tf.nn.softmax(e)
    alphas = alphas * ft_masks
    alphas = alphas / tf.clip_by_value(tf.reduce_sum(alphas, 1, True), 1e-20, 1.)
    _alphas = alphas
    _alphas = tf.reshape(_alphas, (-1, self._config.num_ft, 1))
    phi_V = tf.reduce_sum(_alphas * fts, 1)
    # print phi_V.get_shape(), x.get_shape()

    inputs = tf.concat([x, phi_V], 1)
    out_ops = self.submods[CELL].get_out_ops_in_mode({
      self.submods[CELL].InKey.INPUT: inputs, 
      self.submods[CELL].InKey.STATE: states,
      self.submods[CELL].InKey.IS_TRN: is_trn,
    }, mode)

    return out_ops[self.submods[CELL].OutKey.OUTPUT], out_ops[self.submods[CELL].OutKey.STATE], alphas


def next_step_func_handle(model, Uav_plus_b, fts, ft_masks):
  def next_step_func(wordids, states, outputs, step):
    idx = 0 if step == 0 else 1

    x = tf.nn.embedding_lookup(model.word_embedding_W, wordids)
    outputs, states, alphas = model._attention_on_recurrent(
      x, states, 
      Uav_plus_b[idx], outputs, fts[idx], ft_masks[idx],
      False, framework.model.module.Mode.TST)
    logit = tf.nn.xw_plus_b(outputs, model.softmax_W, model.softmax_B)
    log_prob = tf.nn.log_softmax(logit)
    return log_prob, states, outputs

  return next_step_func

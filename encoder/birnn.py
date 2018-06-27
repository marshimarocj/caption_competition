import enum
import math

import tensorflow as tf
from tensorflow.python.util import nest

import framework.model.module
import framework.impl.cell as cell


CELL = 'cell'
RCELL = 'reverse_cell'


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.subcfgs[CELL] = cell.CellConfig()
    self.subcfgs[RCELL] = cell.CellConfig()

    self.cell_type = 'lstm'
    self.num_step = 10

  def _assert(self):
    assert self.subcfgs[CELL].dim_input == self.subcfgs[RCELL].dim_input
    assert self.subcfgs[CELL].dim_hidden == self.subcfgs[RCELL].dim_hidden


class Encoder(framework.model.module.AbstractModule):
  name_scope = 'birnn.Encoder'

  class InKey(enum.Enum):
    FT = 'ft' # (None, num_step, dim_ft)
    MASK = 'mask' # (None, num_step)
    IS_TRN = 'is_training'
    INIT_STATE = 'init_state'

  class OutKey(enum.Enum):
    OUTPUT = 'output'

  def _set_submods(self):
    if self._config.cell_type == 'gru':
      encoder_cell = cell.GRUCell(self._config.subcfgs[CELL])
      reverse_cell = cell.GRUCell(self._config.subcfgs[RCELL])
    elif self._config.cell_type == 'lstm':
      encoder_cell = cell.LSTMCell(self._config.subcfgs[CELL])
      reverse_cell = cell.LSTMCell(self._config.subcfgs[RCELL])
    reverse_cell.name_scope += '.reverse'

    return {
      CELL: encoder_cell,
      RCELL: reverse_cell,
    }

  def _build_parameter_graph(self):
    pass

  def _steps(self, fts, masks, state, is_training):
    state_size = self.submods[CELL].state_size
    state = [state for _ in nest.flatten(state_size)]
    init_state = nest.pack_sequence_as(state_size, state)

    outputs = []
    state = init_state
    for i in range(self._config.num_step):
      ft = fts[:, i]
      out = self.submods[CELL].get_out_ops_in_mode({
        self.submods[CELL].InKey.INPUT: ft,
        self.submods[CELL].InKey.STATE: state,
        self.submods[CELL].InKey.IS_TRN: is_training,
        }, None)
      output = out[self.submods[CELL].OutKey.OUTPUT] # (None, dim_hidden)
      state = out[self.submods[CELL].OutKey.STATE]
      outputs.append(output)
    outputs = tf.stack(outputs, axis=0)
    outputs = tf.transpose(outputs, perm=[1, 0, 2]) # (None, num_step, dim_hidden)

    rbegin = tf.cast(tf.reduce_sum(masks, axis=1), tf.int32)
    rbegin = tf.ones_like(rbegin)*self._config.num_step - rbegin # (None,)

    routputs = []
    state = init_state
    for i in range(self._config.num_step):
      ft = fts[:, -(i+1)]
      out = self.submods[RCELL].get_out_ops_in_mode({
        self.submods[RCELL].InKey.INPUT: ft,
        self.submods[RCELL].InKey.STATE: state,
        self.submods[RCELL].InKey.IS_TRN: is_training,
        }, None)
      output = out[self.submods[RCELL].OutKey.OUTPUT] # (None, dim_hidden/2)
      idx = tf.ones_like(rbegin)*i
      if self._config.cell_type == 'lstm':
        c = tf.where(idx < rbegin, init_state[0], out[self.submods[RCELL].OutKey.STATE][0])
        h = tf.where(idx < rbegin, init_state[1], out[self.submods[RCELL].OutKey.STATE][1])
        state = (c, h)
      elif self._config.cell_type == 'gru':
        state = tf.where(idx < rbegin, init_state, out[self.submods[RCELL].OutKey.STATE])
      routputs.append(output)
    routputs = tf.stack(routputs, axis=0)[::-1] # (num_step, None, dim_hidden)
    routputs = tf.transpose(routputs, perm=[1, 0, 2]) # (None, num_step, dim_hidden)

    outputs = tf.concat([outputs, routputs], axis=2)

    return outputs

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    with tf.variable_scope(self.name_scope):
      tst_outputs = self._steps(
        in_ops[self.InKey.FT], in_ops[self.InKey.MASK], in_ops[self.InKey.INIT_STATE], in_ops[self.InKey.IS_TRN])
      if mode == framework.model.module.Mode.TRN_VAL:
        return {
          self.OutKey.OUTPUT: outputs,
        }
      else:
        return {
          self.OutKey.OUTPUT: outputs,
        }

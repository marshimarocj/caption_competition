import enum
import math

import tensorflow as tf

import framework.util.expanded_op


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.dim_inputs = []
    self.dim_hidden = 512


class Discriminator(framework.model.module.AbstractModule):
  name_scope = 'rnnve.Adv'

  class InKey(enum.Enum):
    FT = 'fts' # []

  class OutKey(enum.Enum):
    CORR = 'corr'

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    self.inter_fc_Ws = []
    self.inter_fc_Bs = []
    num_group = len(self._config.dim_inputs)
    for i in range(num_group):
      for j in range(i+1, num_group):
        fc_Ws = []
        fc_Bs = []
        dim_inputs = [self._config.dim_inputs[i], self._config.dim_hidden]
        dim_outputs = [self._config.dim_hidden, self._config.dim_inputs[j]]
        for l in range(2):
          fc_W = tf.contrib.framework.model_variable('fc_W_%d_%d_%d'%(i, j, l),
            shape=(dim_inputs[l], dim_outputs[l]), dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
          fc_B = tf.contrib.framework.model_variable('fc_B_%d_%d_%d'%(i, j, l),
            shape=(dim_outputs[l],), dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
          fc_Ws.append(fc_W)
          fc_Bs.append(fc_B)
          self._weights.append(fc_W)
          self._weights.append(fc_B)
        self.inter_fc_Ws.append(fc_Ws)
        self.inter_fc_Bs.append(fc_Bs)

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    with tf.variable_scope(self.name_scope):
      fts = in_ops[self.InKey.FT]
      fts = [framework.util.expanded_op.flip_gradient(ft) for ft in fts]

      num_group = len(fts)
      cnt = 0
      corr = 0.
      for i in range(num_group):
        for j in range(i+1, num_group):
          ft_i = fts[i]
          ft_j = fts[j]

          ft_i = tf.nn.xw_plus_b(ft_i, self.inter_fc_Ws[cnt][0], self.inter_fc_Bs[cnt][0])
          ft_i = tf.nn.relu(ft_i)
          ft_i = tf.nn.xw_plus_b(ft_i, self.inter_fc_Ws[cnt][1], self.inter_fc_Bs[cnt][1])
          ft_i = tf.nn.l2_normalize(ft_i, 1)
          corr_ij = tf.reduce_mean(tf.reduce_sum(ft_i*ft_j, -1))

          corr += corr_ij
          cnt += 1
      corr /= cnt

    return {
      self.OutKey.CORR: corr
    }

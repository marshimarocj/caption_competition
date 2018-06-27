import enum

import tensorflow as tf
import numpy as np

import framework.model.module


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    self.dim_embed = 300
    self.max_words_in_caption = 30

    self.window_sizes = [3, 4, 5]
    self.num_filters = [100, 100, 100]


class Encoder(framework.model.module.AbstractModule):
  name_scope = 'sentence.ConvEncoder'

  class InKey(enum.Enum):
    WVEC = 'wvec'
    IS_TRN = 'is_trn'

  class OutKey(enum.Enum):
    OUTPUT = 'output'

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      self.conv_Ws = []
      self.conv_Bs = []
      for window_size, num_filter in zip(self._config.window_sizes, self._config.num_filters):
        filter_shape = [window_size, self._config.dim_embed, num_filter]
        scale = 1.0 / (self.config.dim_embed*window_size)**0.5
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

  def get_out_ops_in_mode(self, in_ops, mode, **kwargs):
    wvec = in_ops[self.InKey.WVEC]
    is_trn = in_ops[self.InKey.IS_TRN]

    with tf.variable_scope(self.name_scope):
      batch_size = tf.shape(wvec)[0]
      outputs = []
      for i in range(len(self.config.window_sizes)):
        conv_W = self.conv_Ws[i]
        conv_B = self.conv_Bs[i]
        window_size = self._config.window_sizes[i]
        num_filter = self._config.num_filters[i]

        conv_out = tf.nn.conv1d(wvec, conv_W, 1, 'SAME', 
          name='conv_%d'%window_size) # use SAME padding for the ease of mask operation
        conv_out = tf.nn.bias_add(conv_out, conv_B)
        conv_out = tf.layers.dropout(conv_out, 
          noise_shape=(batch_size, 1, num_filter), training=is_trn)
        conv_out = tf.nn.relu(conv_out)
        outputs.append(conv_out)
    return {
      self.OutKey.OUTPUT: outputs
    }

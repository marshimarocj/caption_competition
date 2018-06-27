import enum

import tensorflow as tf
import numpy as np

import framework.model.module


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.dim_embed = 300 # word2vector
    self.num_words = 10870
    self.E = np.empty(0)


class Encoder(framework.model.module.AbstractModule):
  name_scope = 'word.Encoder'

  class InKey(enum.Enum):
    CAPTION = 'caption'

  class OutKey(enum.Enum):
    EMBED = 'embed'

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      if self._config.E.size > 0:
        self.word_embedding_W = tf.contrib.framework.model_variable('word_embedding_W',
          shape=(self._config.num_words, self._config.dim_embed), dtype=tf.float32, 
          initializer=tf.constant_initializer(init_val))
      else:
        self.word_embedding_W = tf.contrib.framework.model_variable('word_embedding_W',
          shape=(self._config.num_words, self._config.dim_embed), dtype=tf.flaot32,
          initializer=tf.random_uniform_initializer())
      self._weights.append(self.word_embedding_W)

  def _embed(self, in_ops):
    captionids = in_ops[self.InKey.CAPTION]
    embed = tf.nn.embedding_lookup(self._weights, captionids)
    return embed

  def get_out_ops_in_mode(self, in_ops, mode, reuse=True, **kwargs):
    with tf.variable_scope(self.name_scope):
      embed_op = self._embed(in_ops)
    return {
      self.OutKey.EMBED: embed_op,
    }

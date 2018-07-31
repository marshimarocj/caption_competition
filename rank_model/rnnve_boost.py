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
import rnnve_orth

WE = rnnve_orth.WE
RNN = rnnve_orth.RNN
CELL = rnnve_orth.CELL
RCELL = rnnve_orth.RCELL


ModelConfig = rnnve_orth.ModelConfig
gen_cfg = rnnve_orth.gen_cfg


class Model(rnnve_orth.Model):
  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      if self._config.loss == 'orth':
        loss = self._outputs[self.OutKey.CORR]
        self.op2monitor['loss'] = loss
      else:

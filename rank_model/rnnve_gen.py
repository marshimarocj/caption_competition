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
import rnnve

WE = rnnve.WE
RNN = rnnve.RNN
CELL = rnnve.CELL
RCELL = rnnve.RCELL

ModelConfig = rnnve.ModelConfig
gen_cfg = rnnve.gen_cfg
Model = rnnve.Model
TrnTst = rnnve.TrnTst
PathCfg = rnnve.PathCfg

class TstReader(framework.model.data.Reader):
  def __init__(self, ft_files, annotation_file, num_candidate):
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.num_candidate = num_candidate

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
    for i, ft in enumerate(self.fts):
      fts = np.expand_dims(ft, 0)
      yield {
        'fts': fts,
        'captionids': self.captionids[i*self.num_candidate:(i+1)*self.num_candidate],
        'caption_masks': self.caption_masks[i*self.num_candidate:(i+1)*self.num_candidate],
      }

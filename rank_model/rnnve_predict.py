import os
import sys
import json
import cPickle
import random
import enum
sys.path.append('../')

import tensorflow as tf
import numpy as np
import sklearn.preprocessing

import framework.model.module
import framework.model.trntst
import framework.model.data
import encoder.birnn
import rnnve
import trntst_util

WE = 'word'
RNN = 'rnn'
CELL = encoder.birnn.CELL
RCELL = encoder.birnn.RCELL

ModelConfig = rnnve.ModelConfig
gen_cfg = rnnve.gen_cfg
Model = rnnve.Model


class TrnTst(framework.model.trntst.TrnTst):
  def predict_in_tst(self, sess, tst_reader, predict_file):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_tst()
    outs = []
    for data in tst_reader.yield_val_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sims = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      out = {}
      out['vid'] = data['vid']
      out['vid_idxs'] = data['vid_idxs']
      out['scores'] = sims[0].tolist()
      outs.append(out)
    with open(predict_file, 'w') as fout:
      json.dump(outs, fout, indent=2)


class TstReader(framework.model.data.Reader):
  def __init__(self, ft_files, annotation_file, pair_file, l2norm=False):
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      if l2norm:
        ft = sklearn.preprocessing.normalize(ft)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    with open(pair_file) as f:
      self.pairs = json.load(f)

  def yield_tst_batch(self, batch_size):
    for pair in self.pairs:
      ft_idx = pair['ft_idx']
      vid = pair['vid']
      caption_idxs = [d[0] for d in pair['caption_idxs']]
      vid_idxs = [d[1] for d in pair['caption_idxs']]
      ft = self.fts[ft_idx]
      fts = np.expand_dims(ft, 0)
      captionids = self.captionids[caption_idxs]
      caption_masks = self.caption_masks[caption_idxs]

      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
        'vid': vid,
        'vid_idxs': vid_idxs,
      }

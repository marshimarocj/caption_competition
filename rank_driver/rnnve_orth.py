import argparse
import sys
import os
import datetime
import cPickle
sys.path.append('../')

import numpy as np

import rank_model.rnnve_orth
import common

WE = rank_model.rnnve_orth.WE
RNN = rank_model.rnnve_orth.RNN
CELL = rank_model.rnnve_orth.CELL
RCELL = rank_model.rnnve_orth.RCELL

def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--memory_fraction', dest='memory_fraction', type=float, default=1.0)
  parser.add_argument('--loss', dest='loss', default='lift')
  # only in tst
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=-1)
  parser.add_argument('--annotation_file', dest='annotation_file')
  parser.add_argument('--out_name', dest='out_name')
  parser.add_argument('--ft_files', dest='ft_files')

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = rank_model.rnnve_orth.PathCfg()
  return common.gen_dir_struct_info(path_cfg, path_cfg_file)

def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = rank_model.rnnve_orth.ModelConfig()
  model_cfg.load(model_cfg_file)

  if path_cfg.embed_file != '':
    E = np.load(path_cfg.embed_file)
    E = E.astype(np.float32)
    model_cfg.subcfgs[WE].E = E
    model_cfg.subcfgs[WE].num_words = E.shape[0]
    model_cfg.subcfgs[WE].dim_embed = E.shape[1]
    model_cfg.subcfgs[RNN].subcfgs[CELL].dim_input = E.shape[1]
    model_cfg.subcfgs[RNN].subcfgs[RCELL].dim_input = E.shape[1]

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  model_cfg.loss = opts.loss
  if opts.is_train and model_cfg.loss == 'orth' : 
    model_cfg.num_epoch = 5
    model_cfg.subcfgs[WE].freeze = True
    model_cfg.subcfgs[RNN].freeze = True
    model_cfg.subcfgs[RNN].subcfgs[CELL].freeze = True
    model_cfg.subcfgs[RNN].subcfgs[RCELL].freeze = True

  m = rank_model.rnnve_orth.Model(model_cfg)

  if opts.is_train:
    if model_cfg.loss == 'orth':
      trntst = rank_model.rnnve_orth.OrthTrnTst(model_cfg, path_cfg, m)
      trn_reader = rank_model.rnnve_orth.OrthReader(
        model_cfg.num_neg, path_cfg.trn_ftfiles, path_cfg.trn_annotation_file)
      val_reader = rank_model.rnnve_orth.OrthReader(
        model_cfg.num_neg, path_cfg.val_ftfiles, path_cfg.val_annotation_file)
    else:
      if opts.best_epoch >= 0:
        path_cfg.model_file = os.path.join(path_cfg.model_dir, 'epoch-%d'%opts.best_epoch)
      trntst = rank_model.rnnve_orth.TrnTst(model_cfg, path_cfg, m)
      trn_reader = rank_model.rnnve_orth.TrnReader(
        model_cfg.num_neg, path_cfg.trn_ftfiles, path_cfg.trn_annotation_file)
      val_reader = rank_model.rnnve_orth.ValReader(
        path_cfg.val_ftfiles, path_cfg.val_annotation_file, path_cfg.val_label_file)

    if path_cfg.model_file != '':
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction, resume=True)
    else:
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction)
  else:
    path_cfg.model_file = os.path.join(path_cfg.model_dir, 'epoch-%d'%opts.best_epoch)
    path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred', '%s.npy'%opts.out_name)
    path_cfg.log_file = None

    trntst = rank_model.rnnve_orth.TrnTst(model_cfg, path_cfg, m)

    tst_reader = rank_model.rnnve_orth.TstReader(opts.ft_files.split(','), opts.annotation_file)
    trntst.test(tst_reader, memory_fraction=opts.memory_fraction)

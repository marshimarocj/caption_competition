import argparse
import sys
import os
import datetime
import cPickle
sys.path.append('../')

import numpy as np

import rank_model.ceve
import common

WE = rank_model.ceve.WE

def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--memory_fraction', dest='memory_fraction', type=float, default=1.0)
  # only in tst
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=True)

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = model.ceve.PathCfg()
  return common.gen_dir_struct_info(path_cfg, path_cfg_file)


def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = rank_model.ceve.ModelConfig()
  model_cfg.load(model_cfg_file)

  E = np.load(path_cfg.embed_file)
  E = E.astype(np.float32)
  model_cfg.subcfgs[WE].E = E
  model_cfg.subcfgs[WE].num_words = E.shape[0]
  model_cfg.subcfgs[WE].dim_embed = E.shape[1]

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  m = rank_model.ceve.Model(model_cfg)

  if opts.is_train:
    trntst = model.rank_ceve.TrnTst(model_cfg, path_cfg, m)

    trn_reader = model.rank_ceve.TrnReader(
      model_cfg.num_neg, path_cfg.trn_ftfiles, path_cfg.trn_annotation_file)
    val_reader = model.rank_ceve.ValReader(
      path_cfg.val_ftfiles, path_cfg.val_annotation_file, path_cfg.val_label_file)
    if path_cfg.model_file != '':
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction, resume=True)
    else:
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction)
  else:
    path_cfg.model_file = os.path.join(path_cfg.model_dir, 'epoch-%d'%opts.best_epoch)
    path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred', 'tst.npy'%opts.best_epoch)
    path_cfg.log_file = None

    trntst = rank_model.ceve.TrnTst(model_cfg, path_cfg, _model)

    tst_reader = rank_model.ceve.TstReader(path_cfg.tst_ftfiles, path_cfg.tst_annotation_file)
    trntst.test(tst_reader, memory_fraction=opts.memory_fraction)

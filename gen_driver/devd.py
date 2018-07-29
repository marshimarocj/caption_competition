import argparse
import sys
import os
import datetime
import cPickle
sys.path.append('../')

import numpy as np

import gen_model.devd
import base


def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--pretrain', dest='pretrain', type=int, default=True)
  parser.add_argument('--memory_fraction', dest='memory_fraction', type=float, default=1.0)
  # only in tst
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=True)
  parser.add_argument('--gen_sent_mode', dest='gen_sent_mode', type=int, default=1,
    help='''
1: top_1 sentence, {vid: caption}, normal output
2: top_k perplexity+sentence {vid: [(perplexity, sent), ...]}, useful for simple ensemble
      '''
  )
  parser.add_argument('--tst_strategy', dest='tst_strategy', default='beam')
  parser.add_argument('--tst_num_sample', dest='tst_num_sample', type=int, default=100)
  parser.add_argument('--tst_sample_topk', dest='tst_sample_topk', type=int, default=10)
  parser.add_argument('--val', dest='val', type=int, default=True)

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = gen_model.devd.PathCfg()
  return base.gen_dir_struct_info(path_cfg, path_cfg_file)


def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = gen_model.devd.ModelConfig()
  model_cfg.load(model_cfg_file)

  # auto fill params
  words = cPickle.load(open(path_cfg.word_file))
  model_cfg.subcfgs[gen_model.devd.VD].num_words = len(words)

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  if opts.is_train:
    m = gen_model.devd.Model(model_cfg)
    with open(os.path.join(path_cfg.log_dir, 'cfg.pkl'), 'w') as fout:
      cPickle.dump(model_cfg, fout)
      cPickle.dump(path_cfg, fout)
      cPickle.dump(opts, fout)

    trntst = gen_model.devd.TrnTst(model_cfg, path_cfg, m)

    trn_reader = gen_model.devd.Reader(
      path_cfg.trn_ftfiles, path_cfg.trn_videoid_file,
      shuffle=True, annotation_file=path_cfg.trn_annotation_file)
    val_reader = gen_model.devd.Reader(
      path_cfg.val_ftfiles, path_cfg.val_videoid_file,
      shuffle=False, annotation_file=path_cfg.val_annotation_file, captionstr_file=path_cfg.groundtruth_file)
    if path_cfg.model_file != '':
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction, resume=True)
    else:
      trntst.train(trn_reader, val_reader, memory_fraction=opts.memory_fraction)
  else:
    path_cfg.model_file = os.path.join(path_cfg.model_dir, 'epoch-%d'%opts.best_epoch)
    path_cfg.log_file = None
    if opts.val:
      path_cfg.tst_ftfiles = path_cfg.val_ftfiles
      path_cfg.tst_videoid_file = path_cfg.val_videoid_file
      out_name = 'val'
    if opts.tst_strategy == 'beam':
      path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred',
        '%s-%d.%d.%d.%s.json'%(out_name,
          opts.best_epoch, opts.gen_sent_mode, model_cfg.subcfgs[gen_model.devd.VD].sent_pool_size, opts.tst_strategy))
    elif opts.tst_strategy == 'sample':
      path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred',
        '%s-%d.%d.%d.%s.json'%(out_name,
          opts.best_epoch, opts.tst_num_sample, opts.tst_sample_topk, opts.tst_strategy))
    path_cfg.log_file = None
    model_cfg.search_strategy = opts.tst_strategy
    model_cfg.tst_num_sample = opts.tst_num_sample
    model_cfg.tst_sample_topk = opts.tst_sample_topk

    m = gen_model.devd.Model(model_cfg)

    trntst = gen_model.devd.TrnTst(model_cfg, path_cfg, m, gen_sent_mode=opts.gen_sent_mode)
    trntst.gen_sent_mode = opts.gen_sent_mode
    tst_reader = gen_model.devd.Reader(
      path_cfg.tst_ftfiles, path_cfg.tst_videoid_file)
    trntst.test(tst_reader, memory_fraction=opts.memory_fraction)

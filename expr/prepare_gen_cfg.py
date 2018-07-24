import os
import sys
import json
sys.path.append('../')

import numpy as np

import gen_model.vevd
import gen_model.self_critique
import gen_model.diversity
import gen_model.margin
import gen_model.vead


'''func
'''
def get_mean_ft_files(root_dir, modal_feature_names, splits, dir_name='mp_feature'):
  dim_fts = []
  split_ftfiles = []
  for split in splits:
    ftfiles = []
    for name in modal_feature_names:
      ftfile = os.path.join(root_dir, dir_name, name, '%s_ft.npy'%split)
      ftfiles.append(ftfile)
      if split == 'val':
        data = np.load(ftfile)
        dim_ft = data.shape[1]

        dim_fts.append(dim_ft)

    split_ftfiles.append(ftfiles)

  return dim_fts, split_ftfiles


def get_att_ft_files(root_dir, model_feature_names, splits, dir_name='sa_feature'):
  split_ft_files = []
  for split in splits:
    ft_files = []
    for name in model_feature_names:
      ft_file = os.path.join(root_dir, dir_name, name, '%s_ft.npz'%split)
      ft_files.append(ft_file)
    split_ft_files.append(ft_files)

  return split_ft_files


'''expr
'''
def prepare_vevd():
  # root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  root_dir = '/data1/jiac/trecvid2018/generation' # mercurial
  annotation_dir = os.path.join(root_dir, 'annotation')
  split_dir = os.path.join(root_dir, 'split')
  splits = ['trn', 'val', 'tst']
  out_dir = os.path.join(root_dir, 'vevd_expr')
  model_spec = 'lstm'

  ft_names = [
    'i3d',
    'resnet200',
  ]

  dim_fts, split_ftfiles = get_mean_ft_files(root_dir, ft_names, splits)

  params = {
    'num_step': 30,
    'dim_input': 512,
    'dim_hidden': 512,
    'num_epoch': 200,
    'content_keepin_prob': 1.,
    'cell_keepin_prob': 0.5,
    'cell_keepout_prob': 0.5,
    'dim_fts': dim_fts,
  }

  model_cfg = gen_model.vevd.gen_cfg(**params)
  outprefix = '%s.%d.%d.%s'%(
    os.path.join(out_dir, '_'.join(ft_names)),
    params['dim_hidden'], params['dim_input'],
    model_spec)
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': split_ftfiles[0],
    'val_ftfiles': split_ftfiles[1],
    'tst_ftfiles': split_ftfiles[2],
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'output_dir': output_dir,
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


def prepare_self_critique():
  # root_dir = '/data1/jiac/trecvid2018/generation' # uranus
  root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
  annotation_dir = os.path.join(root_dir, 'annotation')
  split_dir = os.path.join(root_dir, 'split')
  splits = ['trn', 'val', 'tst']
  out_dir = os.path.join(root_dir, 'self_critique_expr')

  ft_names = [
    'i3d',
    'resnet200',
  ]

  dim_fts, split_ftfiles = get_mean_ft_files(root_dir, ft_names, splits)

  params = {
    'num_epoch': 100,
    'num_sample': 1,
    # 'reward_metric': 'cider',
    'reward_metric': 'bcmr',
    'alpha': 1.,

    'num_step': 30,
    'dim_input': 512,
    'dim_hidden': 512,
    'content_keepin_prob': 1.,
    'cell_keepin_prob': 0.5,
    'cell_keepout_prob': 0.5,
    'dim_fts': dim_fts,
  }

  model_cfg = gen_model.self_critique.gen_cfg(**params)
  outprefix = '%s.%d.%d.%s'%(
    os.path.join(out_dir, '_'.join(ft_names)),
    params['dim_hidden'], params['dim_input'], params['reward_metric'])
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': split_ftfiles[0],
    'val_ftfiles': split_ftfiles[1],
    'tst_ftfiles': split_ftfiles[2],
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


def prepare_diversity():
  root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
  # root_dir = '/data1/jiac/trecvid2018/generation' # uranus
  annotation_dir = os.path.join(root_dir, 'annotation')
  split_dir = os.path.join(root_dir, 'split')
  splits = ['trn', 'val', 'tst']
  out_dir = os.path.join(root_dir, 'diversity_expr')

  ft_names = [
    'i3d',
    'resnet200',
  ]

  dim_fts, split_ftfiles = get_mean_ft_files(root_dir, ft_names, splits)

  params = {
    'num_step': 30,
    'reward_alpha': .25,
    # 'reward_metric': 'cider',
    'reward_metric': 'bcmr',
    'dim_input': 512,
    'dim_hidden': 512,
    'num_epoch': 100,
    'content_keepin_prob': 1.,
    'cell_keepin_prob': 0.5,
    'cell_keepout_prob': 0.5,
    'dim_fts': dim_fts,
    'num_sample': 5,
    'sample_topk': -1,
    'tst_strategy': 'beam',
    'tst_num_sample': '100',
    'tst_sample_topk': 5,
    'min_ngram_in_diversity': 2,
    'max_ngram_in_diversity': 4,
  }

  model_cfg = gen_model.diversity.gen_cfg(**params)
  model_cfg.trn_batch_size = 32
  outprefix = '%s.%d.%d.%.1f.%d.%d_%d.%s'%(
    os.path.join(out_dir, '_'.join(ft_names)),
    params['dim_hidden'], params['dim_input'], params['reward_alpha'], params['num_sample'], 
    params['min_ngram_in_diversity'], params['max_ngram_in_diversity'], params['reward_metric'])
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': split_ftfiles[0],
    'val_ftfiles': split_ftfiles[1],
    'tst_ftfiles': split_ftfiles[2],
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


def prepare_margin():
  root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
  annotation_dir = os.path.join(root_dir, 'annotation')
  split_dir = os.path.join(root_dir, 'split')
  splits = ['trn', 'val', 'tst']
  out_dir = os.path.join(root_dir, 'margin_expr')

  ft_names = [
    'i3d',
    'resnet200',
  ]

  dim_fts, split_ftfiles = get_mean_ft_files(root_dir, ft_names, splits)

  params = {
    'num_step': 30,
    'dim_input': 512,
    'dim_hidden': 512,
    'num_epoch': 100,
    'content_keepin_prob': 1.,
    'cell_keepin_prob': 0.5,
    'cell_keepout_prob': 0.5,
    'dim_fts': dim_fts,

    'reward_alpha': .5,
    'num_neg': 16,
    'num_sample': 5,
    'margin': .1,
    'strategy': 'beam',
    'metric': 'cider',
    # 'metric': 'bcmr',
  }

  model_cfg = gen_model.margin.gen_cfg(**params)
  outprefix = '%s.%d.%d.%.1f.%d.%d.%.1f.%s'%(
    os.path.join(out_dir, '_'.join(ft_names)),
    params['dim_hidden'], params['dim_input'],
    params['reward_alpha'],
    params['num_neg'], params['num_sample'], params['margin'],
    params['metric'])
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': split_ftfiles[0],
    'val_ftfiles': split_ftfiles[1],
    'tst_ftfiles': split_ftfiles[2],
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


def prepare_vead():
  root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
  annotation_dir = os.path.join(root_dir, 'annotation')
  split_dir = os.path.join(root_dir, 'split')
  splits = ['trn', 'val', 'tst']
  out_dir = os.path.join(root_dir, 'vead_expr')

  ft_names = [
    'i3d',
    'resnet200',
  ]

  dim_fts, split_ftfiles = get_mean_ft_files(root_dir, ft_names, splits)
  att_split_ftfiles = get_att_ft_files(root_dir, ft_names, splits)

  params = {
    'num_step': 30,
    'dim_input': 512,
    'dim_hidden': 512,
    'num_epoch': 100,
    'content_keepin_prob': 1.,
    'cell_keepin_prob': 0.5,
    'cell_keepout_prob': 0.5,
    'dim_fts': dim_fts,
    'dim_attention': 512,
    'dim_ft': sum(dim_fts),

    'num_ft': 11,
  }

  model_cfg = gen_model.vead.gen_cfg(**params)
  outprefix = '%s.%d.%d'%(
    os.path.join(out_dir, '_'.join(ft_names)),
    params['dim_hidden'], params['dim_input'])
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': split_ftfiles[0],
    'val_ftfiles': split_ftfiles[1],
    'tst_ftfiles': split_ftfiles[2],
    'trn_att_ftfiles': att_split_ftfiles[0],
    'val_att_ftfiles': att_split_ftfiles[1],
    'tst_att_ftfiles': att_split_ftfiles[2],
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'output_dir': output_dir,
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


if __name__ == '__main__':
  # prepare_vevd()
  # prepare_self_critique()
  # prepare_diversity()
  # prepare_margin()
  prepare_vead()

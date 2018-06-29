import os
import sys
import json
sys.path.append('../')

import numpy as np

import gen_model.vevd


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


'''expr
'''
def prepare_vevd():
  root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
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


if __name__ == '__main__':
  prepare_vevd()

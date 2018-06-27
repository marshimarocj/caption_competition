import os
import sys
import json
sys.path.append('../')

import rank_model.ceve


'''func
'''


'''expr
'''
def prepare_ceve():
  root_dir = '/data1/jiac/trecvid2018/rank' # mercurial
  split_dir = os.path.join(root_dir, 'split')
  label_dir = '/data1/jiac/trecvid2017/label'
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'ceve_expr')
  splits = ['trn', 'val', 'tst']
  
  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 20,

    'alpha': 0.5,
    'num_neg': 32,
    'l2norm': True,
    'dim_ft': 1024 + 2048,
    'dim_joint_embed': 300,

    'window_sizes': [1, 2, 3],
    'pool': 'mean',
  }

  outprefix = '%s.%d.%s.%s.%.1f.%d'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], '_'.join([str(d) for d in window_sizes]), pool,
    alpha, neg2pos)

  model_cfg = rank_model.ceve.gen_cfg(*params)

  model_cfg_file = '%s.model.json'%outprefix
  with open(model_cfg_file, 'w') as fout:
    json.dump(model_cfg, fout, indent=2)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, 'vtt.matching.ranking.set.2.gt'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.A.pkl'),
    'tst_annotation_file': '',
    'word_file': word_file,
    'embed_file': embed_file,
    'output_dir': output_dir,
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


if __name__ == '__main__':
  prepare_ceve()

import os
import sys
import json
sys.path.append('../')

import rank_model.ceve
import rank_model.rnnve
import rank_model.ceve_score
import rank_model.vevd_score
import rank_model.aca
import rank_model.aca_rnn
import rank_model.aca_track
import rank_model.rnnve_feedforward
import rank_model.rnnve_orth


'''func
'''


'''expr
'''
def prepare_ceve():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  # root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'ceve_expr')
  splits = ['trn', 'val', 'tst']
  
  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 50,

    'alpha': 0.5,
    # 'alpha': 1.,
    'num_neg': 32,
    'l2norm': True,
    'dim_ft': 1024 + 2048,
    'dim_joint_embed': 300,

    'max_words_in_caption': 30,
    'window_sizes': [1, 2, 3],
    'num_filters': [100, 100, 100],
    'pool': 'mean',
    # 'pool': 'max',
  }

  outprefix = '%s.%d.%s.%s.%.1f'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], '_'.join([str(d) for d in params['window_sizes']]), 
    params['pool'], params['alpha'])

  model_cfg = rank_model.ceve.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
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


def prepare_ceve_score():
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'ceve_expr')
  splits = ['trn', 'val', 'tst']

  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 50,

    'alpha': 0.5,
    'num_neg': 32,
    'l2norm': True,
    'dim_ft': 1024 + 2048,
    'dim_joint_embed': 300,

    'max_words_in_caption': 30,
    'window_sizes': [1, 2, 3],
    'num_filters': [100, 100, 100],
    'pool': 'max',
  }

  outprefix = '%s.%d.%s.%s.%.1f.score'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], '_'.join([str(d) for d in params['window_sizes']]), 
    params['pool'], params['alpha'])

  model_cfg = rank_model.ceve_score.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.A.pkl'),
    'tst_annotation_file': '',
    'trn_vid_file': os.path.join(split_dir, 'trn_videoids.npy'),
    'word_file': word_file,
    'embed_file': embed_file,
    'output_dir': output_dir,
    'groundtruth_file': os.path.join(root_dir, 'annotation', 'human_caption_dict.pkl'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


def prepare_rnnve():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  # root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  # embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  # embed_file = os.path.join(root_dir, 'annotation', 'E.sbu.word2vec.npy') 
  embed_file = os.path.join(root_dir, 'annotation', 'E.flickr30m.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'rnnve_expr')
  splits = ['trn', 'val', 'tst']
  
  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 100,

    'alpha': 0.5,
    'num_neg': 32,
    'l2norm': True,
    'dim_ft': 1024 + 2048,
    # 'dim_joint_embed': 300,
    # 'dim_joint_embed': 512,
    'dim_joint_embed': 500,

    'max_words_in_caption': 30,
    # 'pool': 'mean',
    'pool': 'max',

    'cell': 'gru',
    # 'cell': 'lstm',
    # 'cell_dim_hidden': 150,
    # 'cell_dim_hidden': 256,
    'cell_dim_hidden': 250,

    'lr_mult': 1,
  }

  # outprefix = '%s.%d.%d.%s.%s.%.1f.sbu'%(
  outprefix = '%s.%d.%d.%s.%s.%.1f.%.1f.flickr30m'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], params['cell_dim_hidden'], params['cell'],
    params['pool'], params['alpha'], params['lr_mult'])

  model_cfg = rank_model.rnnve.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
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


def prepare_vevd_score():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'vevd_expr')
  splits = ['trn', 'val', 'tst']
  model_spec = 'lstm'

  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_step': 30,
    'dim_input': 512,
    'dim_hidden': 512,
    'num_epoch': 100,
    'content_keepin_prob': 1.,
    'cell_keepin_prob': 0.5,
    'cell_keepout_prob': 0.5,
    'dim_fts': [1024, 2048],
    'num_neg': 16,
    'max_margin': 0.5,
  }

  model_cfg = rank_model.vevd_score.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%.1f.%s'%(
    os.path.join(out_dir, '_'.join(ft_names)),
    params['dim_hidden'], params['dim_input'],
    params['num_neg'], params['max_margin'],
    model_spec)
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.A.pkl'),
    'tst_annotation_file': '',
    'trn_vid_file': os.path.join(split_dir, 'trn_videoids.npy'),
    'word_file': word_file,
    'embed_file': embed_file,
    'output_dir': output_dir,
    'groundtruth_file': os.path.join(root_dir, 'annotation', 'human_caption_dict.pkl'),
    'model_file': os.path.join(output_dir, 'model', 'pretrain'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


def prepare_aca():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  # root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  # embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  # embed_file = os.path.join(root_dir, 'annotation', 'E.sbu.word2vec.npy')
  embed_file = os.path.join(root_dir, 'annotation', 'E.flickr30m.word2vec.npy')
  out_dir = os.path.join(root_dir, 'aca_expr')
  splits = ['trn', 'val', 'tst']

  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 50,

    'margin': 0.1,
    'alpha': 0.5,
    'num_neg': 32,
    'dim_ft': 1024 + 2048,
    # 'dim_joint_embed': 300,
    # 'dim_joint_embed': 512,
    'dim_joint_embed': 500,
    'att': True,
    'lr_mult': .1,

    'max_words_in_caption': 30,
  }

  # outprefix = '%s.%d.%.1f.att.sbu'%(
  # outprefix = '%s.%d.%.1f.att.feedforward'%(
  # outprefix = '%s.%d.%.1f.att.flickr30m.feedforward'%(
  outprefix = '%s.%d.%.1f.%.1f.att.flickr30m.feedforward'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], 
    params['alpha'], params['lr_mult'])

  model_cfg = rank_model.aca.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
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


def prepare_aca_rnn():
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'aca_rnn_expr')
  splits = ['trn', 'val', 'tst']

  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 100,

    'margin': 0.1,
    'alpha': 0.5,
    'num_neg': 32,
    'dim_ft': 1024 + 2048,
    'dim_joint_embed': 300,

    'max_words_in_caption': 30,
    'cell': 'gru',
    'cell_dim_hidden': 150,
  }

  outprefix = '%s.%d.%.1f'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], 
    params['alpha'])

  model_cfg = rank_model.aca_rnn.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
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


def prepare_aca_track():
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'aca_track_expr')
  splits = ['trn', 'val', 'tst']

  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 50,

    'margin': 0.1,
    'alpha': 0.5,
    'num_neg': 32,
    'dim_ft': 1024 + 2048,
    'dim_joint_embed': 300,
    'num_track': 11,

    'max_words_in_caption': 30,
  }

  outprefix = '%s.%d.%.1f'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], 
    params['alpha'])

  model_cfg = rank_model.aca_track.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'trn_att_ftfiles': [os.path.join(root_dir, 'sa_feature', ft_name, 'trn_ft.npz') for ft_name in ft_names],
    'val_att_ftfiles': [os.path.join(root_dir, 'sa_feature', ft_name, 'val_ft.2.npz') for ft_name in ft_names],
    'tst_att_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
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


def prepare_rnnve_feedforward():
  # root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.flickr30m.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'rnnve_expr')
  splits = ['trn', 'val', 'tst']
  
  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 100,

    'alpha': 0.5,
    'num_neg': 32,
    'l2norm': True,
    'dim_ft': 1024 + 2048,
    'dim_joint_embed': 512,

    'max_words_in_caption': 30,
    'pool': 'max',
    'dim_word': 500,

    'cell': 'gru',
    'cell_dim_hidden': 512,

    'dim_ft_hiddens': [1024],
    'dim_caption_hiddens': [512],
    'keepin_prob': .8,

    'lr_mult': 1,
  }

  outprefix = '%s.%d.%d.%s.%s.%.1f.%.1f.flickr30m.feedforward'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    params['dim_joint_embed'], params['cell_dim_hidden'], params['cell'],
    params['pool'], params['alpha'], params['lr_mult'])

  model_cfg = rank_model.rnnve_feedforward.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
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


def prepare_rnnve_orth():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  split_dir = os.path.join(root_dir, 'split')
  label_dir = os.path.join(root_dir, 'label')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  embed_file = os.path.join(root_dir, 'annotation', 'E.flickr30m.word2vec.npy') 
  out_dir = os.path.join(root_dir, 'rnnve_orth_expr')
  splits = ['trn', 'val', 'tst']
  
  ft_names = [
    'i3d',
    'resnet200',
  ]

  params = {
    'num_epoch': 100,

    'alpha': 0.5,
    'num_neg': 32,
    'l2norm': True,
    'dim_ft': 1024 + 2048,
    'dim_joint_embeds': [133, 133, 134],

    'max_words_in_caption': 30,
    'pool': 'max',

    'cell': 'gru',
    'cell_dim_hidden': 250,

    'lr_mult': .1,
  }

  outprefix = '%s.%s.%d.%s.%s.%.1f.%.1f.flickr30m'%(
    os.path.join(out_dir, '_'.join(ft_names)), 
    '_'.join([str(d) for d in params['dim_joint_embeds']]),
    params['cell_dim_hidden'], params['cell'],
    params['pool'], params['alpha'], params['lr_mult'])

  model_cfg = rank_model.rnnve.gen_cfg(**params)

  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy') for ft_name in ft_names],
    'val_ftfiles': [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names],
    'tst_ftfiles': [],
    'val_label_file': os.path.join(label_dir, '17.set.2.gt'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.A.pkl'),
    'tst_annotation_file': '',
    'word_file': word_file,
    'embed_file': embed_file,
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, open(path_cfg_file, 'w'), indent=2)


if __name__ == '__main__':
  # prepare_ceve()
  # prepare_rnnve()
  # prepare_ceve_score()
  # prepare_vevd_score()
  # prepare_aca()
  # prepare_aca_rnn()
  # prepare_aca_track()
  # prepare_rnnve_feedforward()
  prepare_rnnve_orth()

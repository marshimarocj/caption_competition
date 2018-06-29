import os
import datetime


def gen_dir_struct_info(path_cfg, path_cfg_file):
  path_cfg.load(path_cfg_file)

  split_dir = path_cfg.split_dir
  output_dir = path_cfg.output_dir
  annotation_dir = path_cfg.annotation_dir

  log_dir = os.path.join(output_dir, 'log')
  if not os.path.exists(log_dir): 
    os.makedirs(log_dir)
  model_dir = os.path.join(output_dir, 'model')
  if not os.path.exists(model_dir): 
    os.makedirs(model_dir)
  predict_dir = os.path.join(output_dir, 'pred')
  if not os.path.exists(predict_dir): 
    os.makedirs(predict_dir)

  path_cfg.log_dir = log_dir
  path_cfg.model_dir = model_dir

  path_cfg.trn_videoid_file = os.path.join(split_dir, 'trn_videoids.npy')
  path_cfg.val_videoid_file = os.path.join(split_dir, 'val_videoids.npy')
  path_cfg.tst_videoid_file = os.path.join(split_dir, 'tst_videoids.npy')

  path_cfg.trn_annotation_file = os.path.join(split_dir, 'trn_id_caption_mask.pkl')
  path_cfg.val_annotation_file = os.path.join(split_dir, 'val_id_caption_mask.pkl')

  path_cfg.groundtruth_file = os.path.join(annotation_dir, 'human_caption_dict.pkl')
  path_cfg.word_file = os.path.join(annotation_dir, 'int2word.pkl')

  timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  path_cfg.log_file = os.path.join(log_dir, 'log-' + timestamp)

  return path_cfg

import os
import json
import subprocess

import numpy as np


'''func
'''
def select_best_epoch(log_dir):
  names = os.listdir(log_dir)
  best_mir = 0.
  best_epoch = -1
  for name in names:
    if 'val_metrics' in name:
      file = os.path.join(log_dir, name)
      with open(file) as f:
        data = json.load(f)
      mir = data['mir']
      epoch = data['epoch']
      if mir > best_mir:
        best_mir = mir
        best_epoch = epoch
  return best_epoch, best_mir


def gen_script_and_run(python_file, model_cfg_file, path_cfg_file, best_epoch, gpuid, **kwargs):
  cmd = [
    'python', python_file,
    model_cfg_file, path_cfg_file, 
    '--is_train', '0',
    '--best_epoch', str(best_epoch),
  ]
  for key in kwargs:
    cmd += ['--' + key, str(kwargs[key])]
  env = os.environ
  env['CUDA_VISIBLE_DEVICES'] = str(gpuid)
  p = subprocess.Popen(cmd, env=env)
  return p


def calc_mir(predicts, vid2gt):
  mir = 0.
  for i, predict in enumerate(predicts):
    idxs = np.argsort(-predict)
    rank = np.where(idxs == vid2gt[i])[0][0]
    rank += 1
    mir += 1. / rank
  mir /= predicts.shape[0]
  return mir


'''expr
'''
def report_best_epoch():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  log_dir = os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.mean.0.5', 'log')
  best_epoch, best_mir = select_best_epoch(log_dir)
  print best_epoch, best_mir


def predict_eval_trecvid17_B():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  ft_names = ['i3d', 'resnet200']
  ft_files = [os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.2.npy') for ft_name in ft_names]
  # annotation_file = os.path.join(root_dir, 'split', 'val_id_caption_mask.B.pkl')
  # out_name = 'val.B'
  annotation_file = os.path.join(root_dir, 'split', 'val_id_caption_mask.A.pkl')
  out_name = 'val.A'
  label_file = os.path.join(root_dir, 'label', '17.set.2.gt')

  expr_name = os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.mean.0.5')
  log_dir = os.path.join(expr_name, 'log')
  model_cfg_file = '%s.model.json'%expr_name
  path_cfg_file = '%s.path.json'%expr_name
  python_file = '../rank_driver/ceve.py'
  # gpuid = 1
  gpuid = 0

  best_epoch, mir_A = select_best_epoch(log_dir)

  p = gen_script_and_run(python_file, model_cfg_file, path_cfg_file, best_epoch, gpuid,
    ft_files=','.join(ft_files), annotation_file=annotation_file, out_name=out_name)
  p.wait()

  # vid2gt = {}
  # with open(label_file) as f:
  #   for line in f:
  #     line = line.strip()
  #     data = line.split(' ')
  #     vid = int(data[0])
  #     gid = int(data[2])
  #     vid2gt[vid] = gid

  # predict_file = '%s/pred/%s.npy'%(expr_name, out_name)
  # predicts = np.load(predict_file)
  # mir_B = calc_mir(predicts, vid2gt)

  # print best_epoch, mir_A, mir_B


if __name__ == '__main__':
  # report_best_epoch()
  predict_eval_trecvid17_B()
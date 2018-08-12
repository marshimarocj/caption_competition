import os
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf

import framework.util.graph_ckpt
import gen_model.vevd


'''func
'''
def select_best_epoch(log_dir):
  names = os.listdir(log_dir)
  best_cider = 0.
  best_epoch = -1
  for name in names:
    if 'val_metrics' in name:
      file = os.path.join(log_dir, name)
      with open(file) as f:
        data = json.load(f)
      cider = data['cider']
      epoch = data['epoch']
      if cider > best_cider:
        best_cider = cider
        best_epoch = epoch
  return best_epoch, best_cider


'''expr
'''
def export_avg_model_weights():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  expr_name = os.path.join(root_dir, 'generation', 'vevd_expr', 'i3d_resnet200_i3d_flow.512.512.lstm')
  log_dir = os.path.join(expr_name, 'log')

  best_epoch, cider = select_best_epoch(log_dir)
  epochs = [best_epoch-10, best_epoch, best_epoch + 10]

  for epoch in epochs:
    model_file = os.path.join(expr_name, 'model', 'epoch-%d'%epoch)
    if not os.path.exists(model_file):
      continue
    name2var = framework.util.graph_ckpt.load_variable_in_ckpt(model_file)
    print name2var.keys()
    break


if __name__ == '__main__':
  export_avg_model_weights()

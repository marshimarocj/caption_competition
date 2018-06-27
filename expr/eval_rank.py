import os
import json


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


'''expr
'''
def report_best_epoch():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  log_dir = os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.mean.0.5', 'log')
  best_epoch, best_mir = select_best_epoch(log_dir)
  print best_epoch, best_mir


if __name__ == '__main__':
  report_best_epoch()

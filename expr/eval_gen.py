import os
import json
import cPickle
import subprocess

import numpy as np

from bleu.bleu import Bleu
from cider.cider import Cider
from rouge.rouge import Rouge
from meteor.meteor import Meteor


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
      mir = data['cider']
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


def get_res_gts_dict(res_file, gts_file):
  human_caption = cPickle.load(file(gts_file))
  data = json.load(file(res_file))

  res, gts = {}, {}
  for key, value in data.iteritems():
    gts[key] = human_caption[int(key)]
    res[key] = [value]

  return res, gts


def eval(predict_file, groundtruth_file):
  res, gts = get_res_gts_dict(predict_file, groundtruth_file)

  bleu_scorer = Bleu(4)
  meteor_scorer = Meteor()
  rouge_scorer = Rouge()
  cider_scorer = Cider()
  # closest score
  res_bleu, _ = bleu_scorer.compute_score(gts, res)
  # metero handles the multi references (don't know the details yet)
  res_meteor, _ = meteor_scorer.compute_score(gts, res)
  meteor_scorer.meteor_p.kill()
  # average
  res_rouge, _ = rouge_scorer.compute_score(gts, res)
  # average
  res_cider, _ = cider_scorer.compute_score(gts, res)

  out = {
    'bleu': res_bleu, 
    'meteor': res_meteor,
    'rouge': res_rouge,
    'cider': res_cider
  }

  return out


'''expr
'''
def predict_eval():
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  # root_dir = '/data1/jiac/trecvid2018/generation' # mercurial
  gt_file = os.path.join(root_dir, 'annotation', 'human_caption_dict.pkl')

  # model_name = 'vevd_expr/i3d_resnet200.512.512.lstm'
  # python_file = '../gen_driver/vevd.py'
  # gpuid = 0

  model_name = 'self_critique/i3d_resnet200.512.512.bcmr'
  python_file = '../gen_driver/self_critique.py'
  gpuid = 0

  log_dir = os.path.join(root_dir, model_name, 'log')
  pred_dir = os.path.join(root_dir, model_name, 'pred')
  model_cfg_file = os.path.join(root_dir, model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, model_name + '.path.json')

  epoch, cider = select_best_epoch(log_dir)

  p = gen_script_and_run(
    python_file, model_cfg_file, path_cfg_file, epoch, 
    gpuid=gpuid)
  p.wait()

  predict_file = os.path.join(pred_dir, 'val-%d.1.5.beam.json'%epoch)
  out = eval(predict_file, gt_file)
  with open('eval.%d.txt'%gpuid, 'w') as fout:
    content = '%.2f\t%.2f\t%.2f'%(
      out['bleu'][3]*100, out['meteor']*100, out['cider']*100)
    print epoch
    print content
    fout.write(str(epoch) + '\t' + content + '\n')


if __name__ == '__main__':
  predict_eval()

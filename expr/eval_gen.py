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


def gen_caption(captionid, words):
  caption = []
  for wid in captionid:
    if wid == 1:
      break
    caption.append(words[wid])
  caption = ' '.join(caption[1:])
  return caption


'''expr
'''
def predict_eval():
  # root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
  # root_dir = '/data1/jiac/trecvid2018/generation' # mercurial
  # root_dir = '/data1/jiac/trecvid2018/generation' # uranus
  gt_file = os.path.join(root_dir, 'annotation', 'human_caption_dict.pkl')

  # model_name = 'vevd_expr/i3d_resnet200.512.512.lstm'
  # # model_name = 'vevd_expr/i3d_i3d_flow.512.512.lstm'
  # # model_name = 'vevd_expr/i3d_resnet200_i3d_flow.512.512.lstm'
  # python_file = '../gen_driver/vevd.py'
  # gpuid = 0

  # model_name = 'devd_expr/i3d_resnet200_i3d_flow.512.512.lstm'
  # python_file = '../gen_driver/devd.py'
  # gpuid = 0

  # # model_name = 'vead_expr/i3d_resnet200.512.512'
  # model_name = 'vead_expr/i3d_resnet200.512.512.context_in_output'
  # python_file = '../gen_driver/vead.py'
  # gpuid = 2

  # model_name = 'vead_expr/i3d_resnet200.512.512'
  # python_file = '../gen_driver/vead.py'
  # gpuid = 1

  # # model_name = 'self_critique_expr/i3d_resnet200.512.512.cider'
  # model_name = 'self_critique_expr/i3d_resnet200.512.512.bcmr'
  # python_file = '../gen_driver/self_critique.py'
  # gpuid = 0

  # # model_name = 'diversity_expr/i3d_resnet200.512.512.0.2.5.2_4.cider'
  # model_name = 'diversity_expr/i3d_resnet200.512.512.0.2.5.2_4.bcmr'
  # python_file = '../gen_driver/diversity.py'
  # gpuid = 0

  # model_name = 'margin_expr/i3d_resnet200.512.512.0.5.16.5.0.1.cider'
  # python_file = '../gen_driver/margin.py'
  # gpuid = 0

  model_name = 'vevd_ensemble_expr/i3d_resnet200.512.512.lstm'
  python_file = '../gen_driver/vevd.py'
  gpuid = 0

  log_dir = os.path.join(root_dir, model_name, 'log')
  pred_dir = os.path.join(root_dir, model_name, 'pred')
  model_cfg_file = os.path.join(root_dir, model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, model_name + '.path.json')

  # epoch, cider = select_best_epoch(log_dir)
  epoch = 200

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


def predict_sample():
  # root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune

  # # model_name = 'diversity_expr/i3d_resnet200.512.512.0.2.5.2_4.cider'
  # model_name = 'diversity_expr/i3d_resnet200.512.512.0.2.5.2_4.bcmr'
  # python_file = '../gen_driver/diversity.py'
  # gpuid = 1

  model_name = 'vevd_expr/i3d_resnet200.512.512.lstm'
  python_file = '../gen_driver/vevd.py'
  gpuid = 0

  log_dir = os.path.join(root_dir, model_name, 'log')
  pred_dir = os.path.join(root_dir, model_name, 'pred')
  model_cfg_file = os.path.join(root_dir, model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, model_name + '.path.json')

  epoch, cider = select_best_epoch(log_dir)

  p = gen_script_and_run(
    python_file, model_cfg_file, path_cfg_file, epoch,
    gpuid=gpuid, tst_strategy='sample', tst_num_sample=100, tst_sample_topk=10, gen_sent_mode=2)
  p.wait()


def rerank_sample():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  model_name = 'rnnve_expr/i3d_resnet200.300.150.gru.max.0.5'
  model_cfg_file = os.path.join(root_dir, 'rank', model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, 'rank', model_name + '.path.json')
  ft_names = ['i3d', 'resnet200']
  ft_files = [os.path.join(root_dir, 'generation', 'mp_feature', ft_name, 'val_ft.npy') for ft_name in ft_names]

  # annotation_file = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'sample.100.pkl')
  # out_file = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'sample.100.npy')

  # annotation_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.cider', 'pred', 'sample.100.pkl')
  # out_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.cider', 'pred', 'sample.100.npy')

  annotation_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'sample.100.pkl')
  out_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'sample.100.npy')

  best_epoch = 77
  num_candidate = 100
  gpuid = 1

  python_file = '../rank_driver/rnnve_gen.py'
  p = gen_script_and_run(
    python_file, model_cfg_file, path_cfg_file, best_epoch, gpuid, 
    annotation_file=annotation_file, ft_files=','.join(ft_files), out_file=out_file, num_candidate=num_candidate)
  p.wait()


def rerank_ensemble():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  model_name = 'rnnve_orth_expr/i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct'
  model_cfg_file = os.path.join(root_dir, 'rank', model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, 'rank', model_name + '.path.json')
  ft_names = ['i3d', 'resnet200']
  ft_files = [os.path.join(root_dir, 'generation', 'mp_feature', ft_name, 'val_ft.npy') for ft_name in ft_names]

  annotation_file = os.path.join(root_dir, 'generation', 'output', 'trecvid17.pkl')
  out_file = os.path.join(root_dir, 'generation', 'output', 'trecvid17.npy')

  best_epoch = 51
  num_candidate = 10
  gpuid = 0

  python_file = '../rank_driver/rnnve_gen.py'
  p = gen_script_and_run(
    python_file, model_cfg_file, path_cfg_file, best_epoch, gpuid, 
    annotation_file=annotation_file, ft_files=','.join(ft_files), out_file=out_file, num_candidate=num_candidate)
  p.wait()


def eval_rerank_caption():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  vid_file = os.path.join(root_dir, 'generation', 'split', 'val_videoids.npy')
  gt_file = os.path.join(root_dir, 'generation', 'annotation', 'human_caption_dict.pkl')
  word_file = os.path.join(root_dir, 'generation', 'annotation', 'int2word.pkl')

  # annotation_file = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'sample.100.pkl')
  # pred_file = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'sample.100.npy')

  # annotation_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.cider', 'pred', 'sample.100.pkl')
  # pred_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.cider', 'pred', 'sample.100.npy')

  # annotation_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'sample.100.pkl')
  # pred_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'sample.100.npy')

  annotation_file = os.path.join(root_dir, 'generation', 'output', 'trecvid17.pkl')
  pred_file = os.path.join(root_dir, 'generation', 'output', 'trecvid17.npy')

  vids = np.load(vid_file)

  with open(gt_file) as f:
    data = cPickle.load(f)
  gts = {}
  for vid in vids:
    gts[vid] = data[vid]

  with open(annotation_file) as f:
    fid, captionids, capiton_masks = cPickle.load(f)
  with open(word_file) as f:
    words = cPickle.load(f)
  scores = np.load(pred_file)
  captionids = np.reshape(captionids, scores.shape + (-1,))
  idxs = np.argmax(scores, 1)
  pred = {}
  for idx, captionid, vid in zip(idxs, captionids, vids):
    pred[vid] = [gen_caption(captionid[idx], words)]
    print vid, gen_caption(captionid[idx], words)

  bleu_scorer = Bleu(4)
  meteor_scorer = Meteor()
  rouge_scorer = Rouge()
  cider_scorer = Cider()
  # closest score
  res_bleu, _ = bleu_scorer.compute_score(gts, pred)
  # metero handles the multi references (don't know the details yet)
  res_meteor, _ = meteor_scorer.compute_score(gts, pred)
  meteor_scorer.meteor_p.kill()
  # average
  res_rouge, _ = rouge_scorer.compute_score(gts, pred)
  # average
  res_cider, _ = cider_scorer.compute_score(gts, pred)

  with open('eval.txt', 'w') as fout:
    content = '%.2f\t%.2f\t%.2f'%(
      res_bleu[3]*100, res_meteor*100, res_cider*100)
    print content
    fout.write(content + '\n')


if __name__ == '__main__':
  # predict_eval()
  # predict_sample()
  # rerank_sample()
  # rerank_ensemble()
  eval_rerank_caption()

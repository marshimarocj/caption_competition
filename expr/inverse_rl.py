import os
import cPickle
import sys
import random
import json
sys.path.append('../')

import numpy as np

from bleu.bleu import Bleu
from rouge.rouge import Rouge
from meteor.meteor import Meteor
from service.fast_cider import CiderScorer, cook_test


'''func
'''
def process_sent(sentence, allow_digits=True):
  sentence = sentence.replace(';', ',')
  sentence = sentence.lower()
  segments = sentence.split(',')
  output = []
  for segment in segments:
    if allow_digits:
      words = re.findall(r"['\-\w]+", segment)
    else:
      words = re.findall(r"['\-A-Za-z]+", segment)
    output.append(' '.join(words))
  output = ' , '.join(output)
  return output


'''expr
'''
def calc_metric_fts():
  root_dir = '/home/jiac/data2/tgif' # gpu9
  caption_file = os.path.join(root_dir, 'aux', 'human_caption_dict.pkl')
  tst_lst_file = os.path.join(root_dir, 'split', 'tst.npy')
  trn_lst_file = os.path.join(root_dir, 'split', 'trn.npy')
  out_file = os.path.join(root_dir, 'inverse_rl', 'metrics.json')

  tst_vids = np.load(tst_lst_file)
  trn_vids = np.load(trn_lst_file)

  bleu_scorer = Bleu(4)
  meteor_scorer = Meteor()
  rouge_scorer = Rouge()
  cider_scorer = CiderScorer()

  with open(caption_file) as f:
    vid2captions = cPickle.load(f)
  cider_scorer.init_refs(vid2captions)

  outs = []
  cnt = 0
  for tst_vid in tst_vids:
    captions = vid2captions[tst_vid]

    num = len(captions)
    for i in range(num):
      gt = {0: [captions[i]]}
      gt_vec, gt_norm, gt_length = cider_scorer._counts2vec(cook_test(gt[0][0]))
      for j in range(num):
        if j == i:
          continue
        pred = {0:[captions[j]]}

        res_bleu, _ = bleu_scorer.compute_score(gt, pred)
        res_meteor, _ = meteor_scorer.compute_score(gt, pred)
        meteor_scorer.meteor_p.kill()
        res_rouge, _ = rouge_scorer.compute_score(gt, pred)

        pred_vec, pred_norm, pred_length = cider_scorer._counts2vec(cook_test(pred[0][0]))
        res_cider = cider_scorer._sim(pred_vec, gt_vec, pred_norm, gt_norm, pred_length, gt_length)        
        res_cider = np.mean(res_cider)
        res_cider *= 10.0

        out.append({
          'pred_id': (tst_vid, j),
          'gt_id': (tst_vid, i),
          'bleu': res_bleu,
          'meteor': res_meteor,
          'rouge': res_rouge,
          'cider': res_cider,
          'label': 1,
        })

        idx =random.randint(0, len(trn_vids-1))
        trn_vid = trn_vids[idx]
        pred = {0: [vid2captions[trn_vid][0]]}

        res_bleu, _ = bleu_scorer.compute_score(gt, pred)
        res_meteor, _ = meteor_scorer.compute_score(gt, pred)
        meteor_scorer.meteor_p.kill()
        res_rouge, _ = rouge_scorer.compute_score(gt, pred)

        pred_vec, pred_norm, pred_length = cider_scorer._counts2vec(cook_test(pred[0][0]))
        res_cider = cider_scorer._sim(pred_vec, gt_vec, pred_norm, gt_norm, pred_length, gt_length)        
        res_cider = np.mean(res_cider)
        res_cider *= 10.0

        out.append({
          'pred_id': (trn_vid, 0),
          'gt_id': (tst_vid, i),
          'bleu': res_bleu,
          'meteor': res_meteor,
          'rouge': res_rouge,
          'cider': res_cider,
          'label': 0,
        })

    cnt += 1
    if cnt % 500 == 0:
      print cnt

  with open(out_file, 'w') as fout:
    json.dump(out, fout, indent=2)


if __name__ == '__main__':
  calc_metric_fts()

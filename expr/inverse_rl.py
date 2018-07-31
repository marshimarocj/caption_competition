import os
import cPickle
import sys
import random
import json
import socket
sys.path.append('../')

import numpy as np

from bleu.bleu import Bleu
from rouge.rouge import Rouge
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
        bleu_scorer = Bleu(4)
        rouge_scorer = Rouge()

        pred = {0:[captions[j]]}

        res_bleu, _ = bleu_scorer.compute_score(gt, pred)
        res_rouge, _ = rouge_scorer.compute_score(gt, pred)

        pred_vec, pred_norm, pred_length = cider_scorer._counts2vec(cook_test(pred[0][0]))
        res_cider = cider_scorer._sim(pred_vec, gt_vec, pred_norm, gt_norm, pred_length, gt_length)        
        res_cider = np.mean(res_cider)
        res_cider *= 10.0
        # print res_bleu, res_rouge, res_cider

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 10000))
        msg = json.dumps([{
          'hyp': pred[0][0],
          'ref': gt[0],
          'id': '0',
        }]) + '\n'
        sock.sendall(msg.encode('utf8'))
        f = sock.makefile()
        line = f.readline()
        line = line.strip()
        id_scores = json.loads(line)
        res_meteor = id_scores[0]['score']
        # print res_meteor
        sock.close()

        outs.append({
          'pred_id': (tst_vid, j),
          'gt_id': (tst_vid, i),
          'bleu': res_bleu,
          'meteor': res_meteor,
          'rouge': res_rouge,
          'cider': res_cider,
          'label': 1,
        })

        bleu_scorer = Bleu(4)
        rouge_scorer = Rouge()

        idx =random.randint(0, len(trn_vids-1))
        trn_vid = trn_vids[idx]
        pred = {0: [vid2captions[trn_vid][0]]}

        res_bleu, _ = bleu_scorer.compute_score(gt, pred)
        res_rouge, _ = rouge_scorer.compute_score(gt, pred)

        pred_vec, pred_norm, pred_length = cider_scorer._counts2vec(cook_test(pred[0][0]))
        res_cider = cider_scorer._sim(pred_vec, gt_vec, pred_norm, gt_norm, pred_length, gt_length)        
        res_cider = np.mean(res_cider)
        res_cider *= 10.0
        # print res_bleu, res_rouge, res_cider

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 10000))
        msg = json.dumps([{
          'hyp': pred[0][0],
          'ref': gt[0],
          'id': '0',
        }]) + '\n'
        sock.sendall(msg.encode('utf8'))
        f = sock.makefile()
        line = f.readline()
        line = line.strip()
        id_scores = json.loads(line)
        res_meteor = id_scores[0]['score']
        # print res_meteor
        sock.close()

        outs.append({
          'pred_id': (trn_vid, 0),
          'gt_id': (tst_vid, i),
          'bleu': res_bleu,
          'meteor': res_meteor,
          'rouge': res_rouge,
          'cider': res_cider,
          'label': 0,
        })

    cnt += 1
    if cnt % 50 == 0:
      print cnt

  with open(out_file, 'w') as fout:
    json.dump(outs, fout, indent=2)


if __name__ == '__main__':
  calc_metric_fts()

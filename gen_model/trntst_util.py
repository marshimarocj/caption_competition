import json

import numpy as np

from bleu.bleu import Bleu
from cider.cider import Cider

import framework.model.trntst

VD = 'decoder'

def predict_and_eval_in_val(trntst, sess, tst_reader, metrics):
  videoid2caption = {}
  base = 0
  op_dict = trntst.model.op_in_val(task='generation')

  batch_size = trntst.model_cfg.tst_batch_size
  for data in tst_reader.yield_tst_batch(batch_size, task='generation'):
    feed_dict = {
      trntst.model.inputs[trntst.model.InKey.FT]: data['fts'],
      trntst.model.inputs[trntst.model.InKey.IS_TRN]: False,
    }
    sent_pool = sess.run(
      op_dict[trntst.model.OutKey.OUT_WID], feed_dict=feed_dict)
    sent_pool = np.array(sent_pool)

    for k, sent in enumerate(sent_pool):
      videoid = tst_reader.videoids[base + k]
      videoid2caption[videoid] = trntst.int2str(np.expand_dims(sent, 0))
    base += batch_size

  bleu_scorer = Bleu(4)
  bleu_score, _ = bleu_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
  for i in range(4):
    metrics['bleu%d'%(i+1)] = bleu_score[i]

  cider_scorer = Cider()
  cider_score, _ = cider_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
  metrics['cider'] = cider_score


def predict_in_tst(trntst, sess, tst_reader, predict_file, search_strategy):
  videoid2caption = {}
  base = 0
  op_dict = trntst.model.op_in_tst()
  for data in tst_reader.yield_tst_batch(trntst.model_cfg.tst_batch_size):
    feed_dict = {
      trntst.model.inputs[trntst.model.InKey.FT]: data['fts'],
      trntst.model.inputs[trntst.model.InKey.IS_TRN]: False,
    }
    if search_strategy == 'greedy':
      sent_pool = sess.run(
        op_dict[trntst.model.OutKey.OUT_WID], feed_dict=feed_dict)
      sent_pool = np.array(sent_pool)

      for k, sent in enumerate(sent_pool):
        videoid = tst_reader.videoids[base + k]
        videoid2caption[videoid] = trntst.int2str(np.expand_dims(sent, 0))[0]

      base += sent_pool.shape[0]
    elif search_strategy == 'beam':
      wordids, cum_log_probs, pres, ends = sess.run(
        [
          op_dict[trntst.model.OutKey.OUT_WID],
          op_dict[trntst.model.OutKey.BEAM_CUM_LOG_PROB],
          op_dict[trntst.model.OutKey.BEAM_PRE],
          op_dict[trntst.model.OutKey.BEAM_END],
        ], feed_dict=feed_dict)
      sent_pool = framework.util.caption.utility.beamsearch_recover_captions(
        wordids, cum_log_probs, pres, ends, trntst.model_cfg.subcfgs[VD].sent_pool_size)

      for b in xrange(len(sent_pool)):
        videoid = str(tst_reader.videoids[b+base])
        output_by_sent_mode(sent_pool[b], videoid, videoid2caption,
          trntst.model_cfg.subcfgs[VD].sent_pool_size, trntst.gen_sent_mode, trntst.int2str)

      base += len(sent_pool)
    elif search_strategy == 'sample':
      out_wids, log_probs = sess.run(
        [op_dict[trntst.model.OutKey.OUT_WID], op_dict[trntst.model.OutKey.LOG_PROB]],
        feed_dict=feed_dict)
      captionids, caption_masks = gen_captionid_masks_from_wids(out_wids)
      caption_masks = caption_masks[:, :, 1:]
      norm_log_prob = np.sum(log_probs * caption_masks, axis=-1) / np.sum(caption_masks, axis=-1) # (None, num_sample)
      idxs = np.argmax(norm_log_prob, axis=1)

      for i in range(captionids.shape[0]):
        videoid = str(tst_reader.videoids[i+base])
        caption = trntst.int2str(captionids[i][idxs[i]:idxs[i]+1])[0]
        videoid2caption[videoid] = caption

      base += captionids.shape[0]

  json.dump(videoid2caption, open(predict_file, 'w'))


def output_by_sent_mode(sent_pool, videoid, videoid2caption,
    sent_pool_size, gen_sent_mode, int2str):
  if gen_sent_mode == 1:
    if len(sent_pool) == 0:
      videoid2caption[videoid] = ''
    else:
      captionid = np.expand_dims(sent_pool[0][1], 0)
      videoid2caption[videoid] = int2str(captionid)[0]
  elif gen_sent_mode == 2:
    videoid2caption[videoid] = []
    for k in xrange(sent_pool_size):
      captionid = np.expand_dims(sent_pool[k][1], 0)
      out = (
        float(sent_pool[k][0]),
        int2str(captionid)[0],
      )
      videoid2caption[videoid].append(out)


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    framework.model.trntst.PathCfg.__init__(self)
    # manually provided in the cfg file
    self.split_dir = ''
    self.annotation_dir = ''
    self.output_dir = ''
    self.trn_ftfiles = []
    self.val_ftfiles = []
    self.tst_ftfiles = []

    # automatically generated paths
    self.trn_videoid_file = ''
    self.val_videoid_file = ''
    self.trn_annotation_file = ''
    self.val_annotation_file = ''
    self.groundtruth_file = ''
    self.word_file = ''


def eval_cider_in_rollout(out_wids, vids, int2str, cider):
  batch_size, num, num_step = out_wids.shape
  out_scores = []
  for i in range(batch_size):
    pred_captions = []
    for j in range(num):
      pred_caption = int2str(np.expand_dims(out_wids[i, j], 0))[0]
      pred_captions.append(pred_caption)
    _vids = [vids[i]]*num
    score, scores = cider.compute_cider(pred_captions, _vids)
    out_scores.append(scores)

  out_scores = np.array(out_scores, dtype=np.float32)
  return out_scores


def gen_captionid_masks_from_wids(out_wids):
  batch_size, num, num_step = out_wids.shape
  bos = np.zeros((batch_size, num, 1), dtype=np.int32)
  caption_ids = np.concatenate([bos, out_wids], axis=2)
  caption_masks = np.zeros(caption_ids.shape, dtype=np.float32)
  caption_masks[:, :, 0] = 1.
  for i in range(1, num_step+1):
    caption_masks[:, :, i] = caption_masks[:, :, i-1] * (caption_ids[:, :, i-1] != 1)

  return caption_ids, caption_masks

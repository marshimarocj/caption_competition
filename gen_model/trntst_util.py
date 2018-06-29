import json

import numpy as np

from bleu.bleu import Bleu
from cider.cider import Cider

VD = 'decoder'

class TrnTst(framework.model.trntst.TrnTst):
  def __init__(self, model_cfg, path_cfg, model, gen_sent_mode=1):
    framework.model.trntst.TrnTst.__init__(self, model_cfg, path_cfg, model)

    # caption int to string
    self.int2str = framework.util.caption.utility.CaptionInt2str(path_cfg.word_file)

    self.gen_sent_mode = gen_sent_mode

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    videoid2caption = {}
    base = 0
    op_dict = self.model.op_in_val(task='generation')

    batch_size = self.model_cfg.tst_batch_size
    for data in tst_reader.yield_tst_batch(batch_size, task='generation'):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sent_pool = sess.run(
        op_dict[self.model.OutKey.OUT_WID], feed_dict=feed_dict)
      sent_pool = np.array(sent_pool)

      for k, sent in enumerate(sent_pool):
        videoid = tst_reader.videoids[base + k]
        videoid2caption[videoid] = self.int2str(np.expand_dims(sent, 0))
      base += batch_size

    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
    for i in range(4):
      metrics['bleu%d'%(i+1)] = bleu_score[i]

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(tst_reader.videoid2captions, videoid2caption)
    metrics['cider'] = cider_score

  def predict_in_tst(self, sess, tst_reader, predict_file):
    videoid2caption = {}
    base = 0
    op_dict = self.model.op_in_tst()
    for data in tst_reader.yield_tst_batch(self.model_cfg.tst_batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      if self.model_cfg.search_strategy == 'greedy':
        sent_pool = sess.run(
          op_dict[self.model.OutKey.OUT_WID], feed_dict=feed_dict)
        sent_pool = np.array(sent_pool)

        for k, sent in enumerate(sent_pool):
          videoid = tst_reader.videoids[base + k]
          videoid2caption[videoid] = self.int2str(np.expand_dims(sent, 0))[0]

        base += sent_pool.shape[0]
      elif self.model_cfg.search_strategy == 'beam':
        wordids, cum_log_probs, pres, ends = sess.run(
          [
            op_dict[self.model.OutKey.OUT_WID],
            op_dict[self.model.OutKey.BEAM_CUM_LOG_PROB],
            op_dict[self.model.OutKey.BEAM_PRE],
            op_dict[self.model.OutKey.BEAM_END],
          ], feed_dict=feed_dict)
        sent_pool = framework.util.caption.utility.beamsearch_recover_captions(
          wordids, cum_log_probs, pres, ends, self.model_cfg.subcfgs[VD].sent_pool_size)

        for b in xrange(len(sent_pool)):
          videoid = str(tst_reader.videoids[b+base])
          output_by_sent_mode(sent_pool[b], videoid, videoid2caption,
            self.model_cfg.subcfgs[VD].sent_pool_size, self.gen_sent_mode, self.int2str)

        base += len(sent_pool)
      elif self.model_cfg.search_strategy == 'sample':
        out_wids, log_probs = sess.run(
          [op_dict[self.model.OutKey.OUT_WID], op_dict[self.model.OutKey.LOG_PROB]],
          feed_dict=feed_dict)
        captionids, caption_masks = gen_captionid_masks_from_wids(out_wids)
        caption_masks = caption_masks[:, :, 1:]
        norm_log_prob = np.sum(log_probs * caption_masks, axis=-1) / np.sum(caption_masks, axis=-1) # (None, num_sample)
        idxs = np.argmax(norm_log_prob, axis=1)

        for i in range(captionids.shape[0]):
          videoid = str(tst_reader.videoids[i+base])
          caption = self.int2str(captionids[i][idxs[i]:idxs[i]+1])[0]
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

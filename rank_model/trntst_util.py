import cPickle
import random
import requests
import threading
from Queue import Queue
import json

import numpy as np

import framework.model.trntst
import framework.util.caption.utility
import service.fast_cider


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    super(PathCfg, self).__init__()
    # manually provided in the cfg file
    self.output_dir = ''
    self.trn_ftfiles = []
    self.val_ftfiles = []
    self.tst_ftfiles = []

    self.val_label_file = ''
    self.word_file = ''
    self.embed_file = ''

    self.trn_annotation_file = ''
    self.val_annotation_file = ''
    self.tst_annotation_file = ''

    # automatically generated paths
    self.log_file = ''


class ScorePathCfg(PathCfg):
  def __init__(self):
    super(ScorePathCfg, self).__init__()
    self.trn_vid_file = ''
    self.groundtruth_file = ''


class AttPathCfg(PathCfg):
  def __init__(self):
    super(AttPathCfg, self).__init__()
    self.trn_att_ftfiles = []
    self.val_att_ftfiles = []
    self.tst_att_ftfiles = []


class TrnTst(framework.model.trntst.TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    mir = 0.
    num = 0
    for data in tst_reader.yield_val_batch(batch_size):
      feed_dict= {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sims = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      idxs = np.argsort(-sims[0])
      rank = np.where(idxs == data['gt'])[0][0]
      rank += 1
      mir += 1. / rank
      num += 1
    mir /= num
    metrics['mir'] = mir

  def predict_in_tst(self, sess, tst_reader, predict_file):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    sims = []
    for data in tst_reader.yield_tst_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sim = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      sims.append(sim)
    sims = np.concatenate(sims, 0)
    np.save(predict_file, sims)


class ScoreTrnTst(TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
      self.model.inputs[self.model.InKey.DELTA]: data['deltas'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }


class AttTrnTst(framework.model.trntst.TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.FT_MASK]: data['ft_masks'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    mir = 0.
    num = 0
    for data in tst_reader.yield_val_batch(batch_size):
      feed_dict= {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.FT_MASK]: data['ft_masks'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sims = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      idxs = np.argsort(-sims[0])
      rank = np.where(idxs == data['gt'])[0][0]
      rank += 1
      mir += 1. / rank
      num += 1
    mir /= num
    metrics['mir'] = mir

  def predict_in_tst(self, sess, tst_reader, predict_file):
    batch_size = self.model_cfg.tst_batch_size
    op_dict = self.model.op_in_val()
    sims = []
    for data in tst_reader.yield_tst_batch(batch_size):
      feed_dict = {
        self.model.inputs[self.model.InKey.FT]: data['fts'],
        self.model.inputs[self.model.InKey.FT_MASK]: data['ft_masks'],
        self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
        self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
        self.model.inputs[self.model.InKey.IS_TRN]: False,
      }
      sim = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      sims.append(sim)
    sims = np.concatenate(sims, 0)
    np.save(predict_file, sims)


class TrnReader(framework.model.data.Reader):
  def __init__(self, num_neg, ft_files, annotation_file):
    self.num_neg = num_neg
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.idxs = []
    self.num_caption = 0

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(fts, axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    self.num_caption = self.captionids.shape[0]
    self.idxs = range(self.num_caption)

  def num_record(self):
    return self.num_caption

  def reset(self):
    random.shuffle(self.idxs)

  def yield_trn_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      pos_idxs = self.idxs[i:i+batch_size]
      pos_ft_idxs = set(self.ft_idxs[pos_idxs].tolist())

      pos_fts = self.fts[self.ft_idxs[pos_idxs]]
      pos_captionids = self.captionids[pos_idxs]
      pos_caption_masks = self.caption_masks[pos_idxs]

      idxs = range(self.num_caption)
      random.shuffle(idxs)

      neg_fts = []
      neg_captionids= []
      neg_caption_masks = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        if ft_idx not in pos_ft_idxs:
          neg_fts.append(self.fts[ft_idx])
          neg_captionids.append(self.captionids[idx])
          neg_caption_masks.append(self.caption_masks[idx])
          if len(neg_fts) == self.num_neg:
            break
      neg_fts = np.array(neg_fts, dtype=np.float32)
      neg_captionids = np.array(neg_captionids, dtype=np.int32)
      neg_caption_masks = np.array(neg_caption_masks, dtype=np.int32)

      fts = np.concatenate([pos_fts, neg_fts], 0)
      captionids = np.concatenate([pos_captionids, neg_captionids], 0)
      caption_masks = np.concatenate([pos_caption_masks, neg_caption_masks], 0)

      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
      }


class ValReader(framework.model.data.Reader):
  def __init__(self, ft_files, annotation_file, label_file):
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.gts = []

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    with open(label_file) as f:
      vid2gid = {}
      for line in f:
        line = line.strip()
        data = line.split(' ')
        vid = int(data[0])
        gid = int(data[1])
        vid2gid[vid] = gid
    for vid in range(len(vid2gid)):
      self.gts.append(vid2gid[vid])

  def yield_val_batch(self, batch_size):
    for ft, gt in zip(self.fts, self.gts):
      fts = np.expand_dims(ft, 0)
      yield {
        'fts': fts,
        'captionids': self.captionids,
        'caption_masks': self.caption_masks,
        'gt': gt,
      }


class TstReader(framework.model.data.Reader):
  def __init__(self, ft_files, annotation_file):
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

  def yield_tst_batch(self, batch_size):
    for ft in self.fts:
      fts = np.expand_dims(ft, 0)
      yield {
        'fts': fts,
        'captionids': self.captionids,
        'caption_masks': self.caption_masks,
      }


class ScoreTrnReader(framework.model.data.Reader):
  def __init__(self, num_neg, ft_files, annotation_file, vid_file, word_file, gt_file):
    self.num_neg = num_neg
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.idxs = []
    self.vids = np.empty(0)
    self.num_caption = 0

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(tuple(fts), axis=1)
    self.fts = self.fts.astype(np.float32)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    self.vids = np.load(vid_file)
    self.int2str = framework.util.caption.utility.CaptionInt2str(word_file)

    self.num_caption = self.captionids.shape[0]
    self.idxs = range(self.num_caption)

    with open(gt_file) as f:
      vid2captions = cPickle.load(f)
    self.cider_scorer = service.fast_cider.CiderScorer()
    self.cider_scorer.init_refs(vid2captions)

  def num_record(self):
    return self.num_caption

  def reset(self):
    random.shuffle(self.idxs)

  def yield_trn_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      pos_idxs = self.idxs[i:i+batch_size]
      pos_ft_idxs = set(self.ft_idxs[pos_idxs].tolist())

      pos_fts = self.fts[self.ft_idxs[pos_idxs]]
      pos_captionids = self.captionids[pos_idxs]
      pos_caption_masks = self.caption_masks[pos_idxs]
      pos_vids = self.vids[self.ft_idxs[pos_idxs]]

      idxs = range(self.num_caption)
      random.shuffle(idxs)

      neg_fts = []
      neg_captionids= []
      neg_caption_masks = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        if ft_idx not in pos_ft_idxs:
          neg_fts.append(self.fts[ft_idx])
          neg_captionids.append(self.captionids[idx])
          neg_caption_masks.append(self.caption_masks[idx])
          if len(neg_fts) == self.num_neg:
            break
      neg_fts = np.array(neg_fts, dtype=np.float32)
      neg_captionids = np.array(neg_captionids, dtype=np.int32)
      neg_caption_masks = np.array(neg_caption_masks, dtype=np.int32)

      neg_captions = self.int2str(neg_captionids)
      deltas = get_scores(neg_captions, pos_vids, self.cider_scorer)

      fts = np.concatenate([pos_fts, neg_fts], 0)
      captionids = np.concatenate([pos_captionids, neg_captionids], 0)
      caption_masks = np.concatenate([pos_caption_masks, neg_caption_masks], 0)

      yield {
        'fts': fts,
        'captionids': captionids,
        'caption_masks': caption_masks,
        'deltas': deltas,
      }


def get_scores(neg_captions, pos_vids, cider_scorer):
  num_neg = len(neg_captions)
  deltas = []
  for pos_vid in pos_vids:
    score, scores = cider_scorer.compute_cider(neg_captions, [pos_vid]*num_neg)
    deltas.append(scores)
  deltas = np.array(deltas, dtype=np.float32)
  deltas = 1.0 - deltas
  deltas = np.maximum(deltas, np.zeros(deltas.shape))

  return deltas


# def cider_scorer(data, q):
#   # server_url = 'http://127.0.0.1:8888/cider'
#   server_url = 'http://172.17.0.1:8888/cider'

#   r = requests.post(server_url, json=data)
#   data = json.loads(r.text)
#   q.put(data)


# def get_scores(neg_captions, pos_vids):
#   num_pos = pos_vids.shape[0]
#   num_neg = len(neg_captions)

#   vid2idx = {}
#   for i, vid in enumerate(pos_vids):
#     vid2idx[vid] = i

#   q = Queue()
#   for vid in pos_vids:
#     eval_data = []
#     for i, neg_caption in enumerate(neg_captions):
#       eval_data.append({
#         'pred': neg_caption,
#         'id': '%d_%d'%(vid, i),
#         'vid': vid,
#       })
#     worker = threading.Thread(
#       target=cider_scorer, args=(eval_data, q))
#     worker.start()

#   deltas = np.zeros((num_pos, num_neg), dtype=np.float32)
#   for t in range(num_pos):
#     data = q.get()
#     # print data['service']
#     data = data['data']
#     for d in data:
#       score = d['score']
#       id = d['id']
#       fields = id.split('_')
#       vid = int(fields[0])
#       j = int(fields[1])
#       deltas[vid2idx[vid], j] = score
#   # print deltas.shape

#   deltas = 1.0 - deltas
#   deltas = np.maximum(deltas, np.zeros(deltas.shape))

#   return deltas


class TrnAttReader(framework.model.data.Reader):
  def __init__(self, num_neg, ft_files, att_ft_files, annotation_file):
    self.num_neg = num_neg
    self.fts = np.empty(0)
    self.ft_masks = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.idxs = []
    self.num_caption = 0

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(fts, axis=1)
    fts = []
    for att_ft_file in att_ft_files:
      data = np.load(att_ft_file)
      ft = data['fts']
      self.ft_masks = data['masks']
      fts.append(ft)
    fts = np.concatenate(fts, axis=2)
    self.fts = np.expand_dims(self.fts, 1)
    self.fts = np.concatenate([self.fts, fts], axis=1)
    self.fts = self.fts.astype(np.float32)
    mask = np.ones((self.fts.shape[0], 1), dtype=np.float32)
    self.ft_masks = np.concatenate([mask, self.ft_masks], 1)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    self.num_caption = self.captionids.shape[0]
    self.idxs = range(self.num_caption)

  def num_record(self):
    return self.num_caption

  def reset(self):
    random.shuffle(self.idxs)

  def yield_trn_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      pos_idxs = self.idxs[i:i+batch_size]
      pos_ft_idxs = set(self.ft_idxs[pos_idxs].tolist())

      pos_fts = self.fts[self.ft_idxs[pos_idxs]]
      pos_ft_masks = self.ft_masks[self.ft_idxs[pos_idxs]]
      pos_captionids = self.captionids[pos_idxs]
      pos_caption_masks = self.caption_masks[pos_idxs]

      idxs = range(self.num_caption)
      random.shuffle(idxs)

      neg_fts = []
      neg_ft_masks = []
      neg_captionids= []
      neg_caption_masks = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        if ft_idx not in pos_ft_idxs:
          neg_fts.append(self.fts[ft_idx])
          neg_ft_masks.append(self.ft_masks[ft_idx])
          neg_captionids.append(self.captionids[idx])
          neg_caption_masks.append(self.caption_masks[idx])
          if len(neg_fts) == self.num_neg:
            break
      neg_fts = np.array(neg_fts, dtype=np.float32)
      neg_ft_masks = np.array(neg_ft_masks, dtype=np.float32)
      neg_captionids = np.array(neg_captionids, dtype=np.int32)
      neg_caption_masks = np.array(neg_caption_masks, dtype=np.int32)

      fts = np.concatenate([pos_fts, neg_fts], 0)
      ft_masks = np.concatenate([pos_ft_masks, neg_ft_masks], 0)
      captionids = np.concatenate([pos_captionids, neg_captionids], 0)
      caption_masks = np.concatenate([pos_caption_masks, neg_caption_masks], 0)

      yield {
        'fts': fts,
        'ft_masks': ft_masks,
        'captionids': captionids,
        'caption_masks': caption_masks,
      }


class ValAttReader(framework.model.data.Reader):
  def __init__(self, ft_files, att_ft_files, annotation_file, label_file):
    self.fts = np.empty(0)
    self.ft_masks = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)
    self.gts = []

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(fts, axis=1)
    fts = []
    for att_ft_file in att_ft_files:
      data = np.load(att_ft_file)
      ft = data['fts']
      self.ft_masks = data['masks']
      fts.append(ft)
    fts = np.concatenate(fts, axis=2)
    self.fts = np.expand_dims(self.fts, 1)
    self.fts = np.concatenate([self.fts, fts], axis=1)
    self.fts = self.fts.astype(np.float32)
    mask = np.ones((self.fts.shape[0], 1), dtype=np.float32)
    self.ft_masks = np.concatenate([mask, self.ft_masks], 1)
    # print self.fts.shape

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

    with open(label_file) as f:
      vid2gid = {}
      for line in f:
        line = line.strip()
        data = line.split(' ')
        vid = int(data[0])
        gid = int(data[1])
        vid2gid[vid] = gid
    for vid in range(len(vid2gid)):
      self.gts.append(vid2gid[vid])

  def yield_val_batch(self, batch_size):
    for ft, ft_mask, gt in zip(self.fts, self.ft_masks, self.gts):
      fts = np.expand_dims(ft, 0)
      ft_masks = np.expand_dims(ft_mask, 0)
      yield {
        'fts': fts,
        'ft_masks': ft_masks,
        'captionids': self.captionids,
        'caption_masks': self.caption_masks,
        'gt': gt,
      }


class TstAttReader(framework.model.data.Reader):
  def __init__(self, ft_files, att_ft_files, annotation_file):
    self.fts = np.empty(0)
    self.ft_masks = np.empty(0)
    self.ft_idxs = np.empty(0)
    self.captionids = np.empty(0)
    self.caption_masks = np.empty(0)

    fts = []
    for ft_file in ft_files:
      ft = np.load(ft_file)
      fts.append(ft)
    self.fts = np.concatenate(fts, axis=1)
    fts = []
    for att_ft_file in att_ft_files:
      data = np.load(att_ft_file)
      ft = data['fts']
      self.ft_masks = data['masks']
      fts.append(ft)
    fts = np.concatenate(fts, axis=2)
    self.fts = np.expand_dims(self.fts, 1)
    self.fts = np.concatenate([self.fts, fts], axis=1)
    self.fts = self.fts.astype(np.float32)
    mask = np.ones((self.fts.shape[0], 1), dtype=np.float32)
    self.ft_masks = np.concatenate([mask, self.ft_masks], 1)

    data = cPickle.load(file(annotation_file))
    self.ft_idxs = data[0]
    self.captionids = data[1]
    self.caption_masks = data[2]

  def yield_tst_batch(self, batch_size):
    for ft, ft_mask in zip(self.fts, self.ft_masks):
      fts = np.expand_dims(ft, 0)
      ft_masks = np.expand_dims(ft_mask, 0)
      yield {
        'fts': fts,
        'ft_masks': ft_masks,
        'captionids': self.captionids,
        'caption_masks': self.caption_masks,
      }

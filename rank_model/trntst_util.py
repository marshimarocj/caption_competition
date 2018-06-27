import cPickle
import random

import numpy as np

import framework.model.trntst


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


class TrnTst(framework.model.trntst.TrnTst):
  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.CAPTIONID]: data['captionids'],
      self.model.inputs[self.model.InKey.CAPTION_MASK]: data['caption_masks'],
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
    self.fts = np.concatenate(tuple(fts), axis=1)
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

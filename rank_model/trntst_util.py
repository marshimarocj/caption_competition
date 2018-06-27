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
      self.model.inputs[self.InKey.FT]: data['fts'],
      self.model.inputs[self.InKey.CAPTIONID]: data['captionids'],
      self.model.inputs[self.InKey.CAPTION_MASK]: data['caption_masks'],
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
      rank = np.where(idxs == gt)[0][0]
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
        self.model.InKey.FT: data['fts'],
        self.model.InKey.CAPTIONID: data['captionids'],
        self.model.InKey.CAPTION_MASK: data['caption_masks'],
      }
      sim = sess.run(op_dict[self.model.OutKey.SIM], feed_dict=feed_dict)
      sims.append(sim)
    sims = np.concatenate(sims, 0)
    np.save(predict_file, sims)

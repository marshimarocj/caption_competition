import os
import cPickle
import json

import numpy as np

from prepare import caption2id_mask


'''func
'''


'''expr
'''
def gen_captionid_mask():
  # root_dir = '/mnt/data1/jiac/trecvid2018/generation' # neptune
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune

  # pred_dir = os.path.join(root_dir, 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.cider', 'pred')
  # caption_file = os.path.join(pred_dir, 'val-51.100.10.sample.json')

  # pred_dir = os.path.join(root_dir, 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred')
  # caption_file = os.path.join(pred_dir, 'val-89.100.10.sample.json')

  pred_dir = os.path.join(root_dir, 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred')
  caption_file = os.path.join(pred_dir, 'val-136.100.10.sample.json')

  vid_file = os.path.join(root_dir, 'split', 'val_videoids.npy')
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  out_file = os.path.join(pred_dir, 'sample.100.pkl')

  max_num_words_in_caption = 30

  with open(caption_file) as f:
    vid2captions = json.load(f)

  vids = np.load(vid_file)

  word2id = {}
  with open(word_file) as f:
    data = cPickle.load(f)
    for i, d in enumerate(data):
      word2id[d] = i

  ft_idxs = []
  captionids = []
  caption_masks = []
  for i, vid in enumerate(vids):
    captions = [d[1] for d in vid2captions[str(vid)]]
    for caption in captions:
      captionid, caption_mask = caption2id_mask(caption, max_num_words_in_caption, word2id)
      ft_idxs.append(i)
      captionids.append(captionid)
      caption_masks.append(caption_mask)
  ft_idxs = np.array(ft_idxs, dtype=np.int32)
  captionids = np.array(captionids, dtype=np.int32)
  caption_masks = np.array(caption_masks, dtype=np.int32)
  with open(out_file, 'w') as fout:
    cPickle.dump([ft_idxs, captionids, caption_masks], fout)


def format_caption():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  out_root_dir = os.path.join(root_dir, 'generation', 'output')

  # pred_file = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'viz.json')
  # out_dir = os.path.join(out_root_dir, 'vevd', 'val17')
  # pred_file = os.path.join(root_dir, 'generation', 'vevd_ensemble_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'viz.json')
  # out_dir = os.path.join(out_root_dir, 'vevd_ensemble', 'val17')
  # pred_file = os.path.join(root_dir, 'generation', 'self_critique_expr', 'i3d_resnet200.512.512.bcmr', 'pred', 'viz.json')
  # out_dir = os.path.join(out_root_dir, 'self_critique', 'val17')
  pred_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'viz.json')
  out_dir = os.path.join(out_root_dir, 'diversity', 'val17')
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  out = {}
  with open(pred_file) as f:
    data = json.load(f)
    for d in data:
      vid = d['vid']
      caption = d['caption']
      name = 'trecvid17_%d.mp4'%vid
      out[name] = [caption]

  # out_file = os.path.join(out_dir, 'epoch.200.json')
  # out_file = os.path.join(out_dir, 'epoch.88.json')
  out_file = os.path.join(out_dir, 'epoch.89.json')
  with open(out_file, 'w') as fout:
    json.dump(out, fout, indent=2)


if __name__ == '__main__':
  # gen_captionid_mask()
  format_caption()

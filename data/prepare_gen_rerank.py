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
  # pred_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'viz.json')
  # out_dir = os.path.join(out_root_dir, 'diversity', 'val17')
  pred_file = os.path.join(root_dir, 'generation', 'margin_expr', 'i3d_resnet200.512.512.0.5.16.5.0.1.cider', 'pred', 'viz.json')
  out_dir = os.path.join(out_root_dir, 'margin', 'val17')
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
  # out_file = os.path.join(out_dir, 'epoch.89.json')
  out_file = os.path.join(out_dir, 'epoch.200.json')
  with open(out_file, 'w') as fout:
    json.dump(out, fout, indent=2)


def gen_captionid_mask_ensemble():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  vid_file = os.path.join(root_dir, 'generation', 'split', 'val_videoids.npy')
  word_file = os.path.join(root_dir, 'generation', 'annotation', 'int2word.pkl')
  out_file = os.path.join(root_dir, 'generation', 'trecvid17.pkl')

  pred_files = [
    os.path.join(root_dir, 'generation', 'output', 'margin', 'val17', 'epoch.200.json'),
    os.path.join(root_dir, 'generation', 'output', 'diversity', 'val17', 'epoch.89.json'),
    os.path.join(root_dir, 'generation', 'output', 'vevd_ensemble', 'val17', 'epoch.200.json'),
    os.path.join(root_dir, 'generation', 'output', 'vevd', 'val17', 'epoch.136.json'),
    os.path.join(root_dir, 'generation', 'output', 'attn', 'val17', 'epoch.35.json'),
    os.path.join(root_dir, 'generation', 'output', 'attn.sc.cider', 'val17', 'epoch.97.json'),
    os.path.join(root_dir, 'generation', 'output', 'audio+', 'val17', 'epoch.36.json'),
    os.path.join(root_dir, 'generation', 'output', 'audio+sc.cider', 'val17', 'epoch.43.json'),
    os.path.join(root_dir, 'generation', 'output', 'ensemble', 'val17', 'ensemble.f0.json'),
    os.path.join(root_dir, 'generation', 'output', 'attn+trecvid16.sc.cider', 'val17', 'epoch.5.json'),
  ]

  max_num_words_in_caption = 30

  word2id = {}
  with open(word_file) as f:
    data = cPickle.load(f)
    for i, d in enumerate(data):
      word2id[d] = i

  ft_idxs = []
  captionids = []
  caption_masks = []
  for pred_file in pred_files:
    with open(pred_file) as f:
      data = json.load(f)
    for key in data:
      name, _ = os.path.splitext(key)
      pos = name.find('_')
      vid = int(name[pos+1:])
      idx = vid-1
      caption = data[key]

      captionid, caption_mask = caption2id_mask(caption, max_num_words_in_caption, word2id)

      ft_idxs.append(idx)
      captionids.append(captionid)
      caption_masks.append(caption_mask)

  ft_idxs = np.array(ft_idxs, dtype=np.int32)
  captionids = np.array(captionids, dtype=np.int32)
  caption_masks = np.array(caption_masks, dtype=np.int32)
  with open(out_file, 'w') as fout:
    cPickle.dump([ft_idxs, captionids, caption_masks], fout)


if __name__ == '__main__':
  # gen_captionid_mask()
  # format_caption()
  gen_captionid_mask_ensemble()

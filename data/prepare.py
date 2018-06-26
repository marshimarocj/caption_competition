import os
import cPickle
import json

import numpy as np


'''func
'''
def caption2id_mask(caption, max_words_in_caption, word2int):
  words = caption.split(' ')
  captionid = np.ones((max_words_in_caption,), dtype=np.int32)
  mask = np.zeros((max_words_in_caption,), dtype=np.int32)
  captionid[0] = 0
  mask[0] = 1
  for i, word in enumerate(words):
    if word in word2int:
      wid = word2int[word]
    else:
      wid = 2
    captionid[i+1] = wid
    mask[i+1] = 1

    if i+1 == max_words_in_caption-1:
      break
  i += 1
  if i+1 < max_words_in_caption:
    captionid[i+1] = 1
    mask[i+1] = 1
  return captionid, mask


'''expr
'''
def merge_tgif_trecvid16_rank_trn():
  trecvid_root_dir = '/data1/jiac/trecvid2017/rank' # mercurial
  tgif_root_dir = '/data1/jiac/tgif'
  out_root_dir = '/data1/jiac/trecvid2018/rank'

  ##########ft#########
  # tgif_ft_files = [
  #   'trn_ft.npy',
  #   'val_ft.npy',
  #   'tst_ft.npy',
  # ]
  # trecvid_ft_files = [
  #   'tst_ft.npy',
  # ]
  # for ft_name in ['i3d', 'resnet200']:
  #   print ft_name
  #   fts = []
  #   for tgif_ft_file in tgif_ft_files:
  #     file = os.path.join(tgif_root_dir, 'mp_feature', ft_name, tgif_ft_file)
  #     ft = np.load(file)
  #     fts.append(ft)
  #   for trecivd_ft_file in trecvid_ft_files:
  #     file = os.path.join(trecvid_root_dir, 'mp_feature', ft_name, trecivd_ft_file)
  #     ft = np.load(file)
  #     fts.append(ft)
  #   fts = np.concatenate(fts, 0)
  #   out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'trn_ft.npy')
  #   np.save(out_file, fts)

  #########caption mask ##########
  # tgif_caption_mask_files = [
  #   os.path.join(tgif_root_dir, 'split', 'trn_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'val_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'tst_id_caption_mask.pkl'),
  # ]
  # trecvid_caption_mask_files = [
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.A.pkl'),
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.B.pkl'),
  # ]
  # idxs = []
  # caption_ids = []
  # caption_masks = []
  # base = 0
  # for file in tgif_caption_mask_files:
  #   with open(file) as f:
  #     data = cPickle.load(f)
  #   idxs.append(data[0] + base)
  #   caption_ids.append(data[1])
  #   caption_masks.append(data[2])
  #   base = np.max(data[0] + base) + 1
  # for file in trecvid_caption_mask_files:
  #   with open(file) as f:
  #     data = cPickle.load(f)
  #   idxs.append(data[0] + base)
  #   caption_ids.append(data[1])
  #   caption_masks.append(data[2])
  # idxs = np.concatenate(idxs, 0)
  # caption_ids = np.concatenate(caption_ids, 0)
  # caption_masks = np.concatenate(caption_masks, 0)
  # out_file = os.path.join(out_root_dir, 'split', 'trn_id_caption_mask.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump([idxs, caption_ids, caption_masks], fout)

  ##########vid##########
  # tgif_vid_files = [
  #   os.path.join(tgif_root_dir, 'split', 'trn_videoids.npy'),
  #   os.path.join(tgif_root_dir, 'split', 'val_videoids.npy'),
  #   os.path.join(tgif_root_dir, 'split', 'tst_videoids.npy'),
  # ]
  # vids = []
  # for file in tgif_vid_files:
  #   vids.append(np.load(file))
  # vids = np.concatenate(vids, 0)
  # base = np.max(vids) + 1
  # vids = [vids, range(base, base + 1915)]
  # vids = np.concatenate(vids)
  # out_file = os.path.join(out_root_dir, 'split', 'trn_videoids.npy')
  # np.save(out_file, vids)

  ##########caption_dict##########
  tgif_caption_file = os.path.join(tgif_root_dir, 'annotation', 'human_caption_dict.pkl')
  trecvid_caption_file = os.path.join(trecvid_root_dir, 'annotation', 'human_caption_dict.pkl')
  vid2captions = {}
  with open(tgif_caption_file) as f:
    data = cPickle.load(f)
  vid2captions.update(data)
  base = len(vid2captions)
  with open(trecvid_caption_file) as f:
    data = cPickle.load(f)
  start = len(data) - 1915
  for vid in range(start, start + 1915):
    vid2captions[vid - start + base] = data[vid]
  out_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump(vid2captions, fout)


def prepare_trecvid17_gen_val():
  trecvid_root_dir = '/data1/jiac/trecvid2017' # mercurial
  out_root_dir = '/data1/jiac/trecvid2018/rank'
  word_file = os.path.join(trecvid_root_dir, 'rank', 'annotation', 'int2word.pkl')
  gt_file = os.path.join(trecvid_root_dir, 'label', 'description', 'trecvid17.json')

  max_words_in_caption = 30

  word2int = {}
  with open(word_file) as f:
    words = cPickle.load(f)
  for i, word in enumerate(words):
    word2int[word] = i

  with open(gt_file) as f:
    data = json.load(f)
  num = len(data)
  idxs = []
  captionids = []
  caption_masks = []
  for vid in range(1, num+1):
    captions = data[str(vid)]
    for caption in captions:
      captionid, caption_mask = caption2id_mask(caption, max_words_in_caption, word2int)

      idxs.append(vid-1)
      captionids.append(captionid)
      caption_masks.append(mask)
  idxs = np.array(idxs, dtype=np.int32)
  captionids = np.array(captionids, dtype=np.int32)
  caption_masks = np.array(caption_masks, dtype=np.int32)

  out_file = os.path.join(out_root_dir, 'split', 'val_id_caption_mask.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump([idxs, captionids, caption_masks], fout)

  # vid2captions = {}
  # caption_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pk')
  # with open(caption_file) as f:
  #   vid2captions = cPickle.load(f)
  # base = 0
  # for vid in vid2captions:
  #   if vid > base:
  #     base = vid
  # base += 1

  # with open(gt_file) as f:
  #   data = json.load(gt_file)
  # for d in data:
  #   vid = int(d['image_id'])-1
  #   vid += base
  #   if vid not in vid2captions:
  #     vid2captions[vid] = []
  #   vid2captions[vid].append(caption)
  # out_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump(vid2captions, fout)


if __name__ == '__main__':
  # merge_tgif_trecvid16_rank_trn()
  prepare_trecvid17_gen_val()

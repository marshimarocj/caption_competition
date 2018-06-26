import os
import cPickle

import numpy as np


'''func
'''


'''expr
'''
def merge_tgif_trecvid16_rank_trn():
  trecvid_root_dir = '/data1/jiac/trecvid2017/rank' # mercurial
  tgif_root_dir = '/data1/jiac/tgif'
  out_root_dir = '/data1/jiac/trecvid2018/rank'

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

  tgif_caption_file = os.path.join(tgif_root_dir, 'annotation', 'human_caption_dict.pkl')
  trecvid_caption_file = os.path.join(trecvid_root_dir, 'annotation', 'human_capiton_dict.pkl')
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


def prepare_trecvid17_rank_val():
  root_dir = '' #


if __name__ == '__main__':
  merge_tgif_trecvid16_rank_trn()

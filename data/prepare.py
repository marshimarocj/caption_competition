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

  tgif_caption_mask_files = [
    os.path.join(tgif_root_dir, 'split', 'trn_id_caption_mask.pkl'),
    os.path.join(tgif_root_dir, 'split', 'val_id_caption_mask.pkl'),
    os.path.join(tgif_root_dir, 'split', 'tst_id_caption_mask.pkl'),
  ]
  trecvid_caption_mask_files = [
    os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.A.pkl'),
    os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.B.pkl'),
  ]
  idxs = []
  caption_ids = []
  caption_masks = []
  base = 0
  for file in tgif_caption_mask_files:
    with open(file) as f:
      data = cPickle.load(f)
    idxs.append(data[0] + base)
    caption_ids.append(data[1])
    caption_masks.append(data[2])
    base = np.max(data[0] + base)
  for file in trecvid_caption_mask_files:
    with open(file) as f:
      data = cPickle.load(f)
    idxs.append(data[0] + base)
    caption_ids.append(data[1])
    caption_masks.append(data[2])
  idxs = np.concatenate(idxs, 0)
  caption_ids = np.concatenate(caption_ids, 0)
  caption_masks = np.concatenate(caption_masks, 0)
  out_file = os.path.join(out_root_dir, 'split', 'trn_id_caption_mask.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump([idxs, caption_ids, caption_masks], fout)


def prepare_trecvid17_rank_val():
  root_dir = '' #


if __name__ == '__main__':
  merge_tgif_trecvid16_rank_trn()

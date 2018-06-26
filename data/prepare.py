import os
import cPickle

import numpy as np


'''func
'''


'''expr
'''
def merge_tgif_trecvid16_rank_trn():
  trecvid_root_dir = '/data1/jiac/trecvid2017' # mercurial
  tgif_root_dir = '/data1/jiac/tgif'
  out_root_dir = '/data1/jiac/trecvid2018'

  tgif_ft_files = [
    'trn_ft.npy',
    'val_ft.npy',
    'tst_ft.npy',
  ]
  trecvid_ft_files = [
    'tst_ft.npy',
  ]
  for ft_name in ['i3d', 'resnet200']:
    print ft_name
    fts = []
    for tgif_ft_file in tgif_ft_files:
      file = os.path.join(tgif_root_dir, 'mp_feature', ft_name, tgif_ft_file)
      ft = np.load(file)
      fts.append(ft)
    fts = np.concatenate(fts, 0)
    out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'trn_ft.npy')
    np.save(out_file, fts)

  # tgif_caption_mask_files = [
  #   os.path.join(tgif_root_dir, 'split', 'trn_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'val_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'tst_id_caption_mask.pkl'),
  # ]
  # trecvid_caption_mask_files = [
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.A.pkl'),
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.B.pkl'),
  # ]



def prepare_trecvid17_rank_tst():
  root_dir = '' # 


if __name__ == '__main__':
  merge_tgif_trecvid16_rank_trn()

import os
import cPickle
import json

import numpy as np


'''func
'''


'''expr
'''
def merge_tgif_trecvid16_trn_track_ft():
  trecvid_root_dir = '' # gpu8
  tgif_root_dir = '/home/jiac/data/tgif/'
  out_root_dir = '/home/jiac/data/trecvid2018/rank'

  tgif_video_name_file = os.path.join(tgif_root_dir, 'aux', 'int2video.npy')
  video_names = np.load(tgif_video_name_file)
  vid_files = [
    os.path.join(tgif_root_dir, 'split', 'trn_videoids.npy'),
    os.path.join(tgif_root_dir, 'split', 'val_videoids.npy'),
    os.path.join(tgif_root_dir, 'split', 'tst_videoids.npy'),
  ]
  tgif_ft_file = os.path.join(tgif_root_dir, 'track_ft', 'i3d_rgb')

  num_tracks = []
  for vid_file in vid_files:
    vids = np.load(vid_file)
    for vid in vids:
      video = video_names[vid]
      ft_file = os.path.join(tgif_ft_file, video + '.npy')
      if os.path.exists(ft_file):
        fts = np.load(ft_file)
        num_tracks.append(fts.shape[0])
  print np.mean(num_tracks), np.median(num_tracks), np.percentile(num_tracks, 90)


if __name__ == '__main__':
  merge_tgif_trecvid16_trn_track_ft()

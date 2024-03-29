import os
import cPickle
import json

import numpy as np


'''func
'''


'''expr
'''
def merge_tgif_trecvid16_trn_track_ft():
  trecvid_root_dir = '/home/jiac/data/trecvid' # gpu8
  tgif_root_dir = '/home/jiac/data/tgif/'
  out_root_dir = '/home/jiac/data/trecvid2018'

  # ft_name = 'i3d_rgb'
  # out_ft_name = 'i3d'
  # dim_ft = 1024
  ft_name = 'resnet200'
  out_ft_name = 'resnet200'
  dim_ft = 2048

  tgif_video_name_file = os.path.join(tgif_root_dir, 'aux', 'int2video.npy')
  video_names = np.load(tgif_video_name_file)
  vid_files = [
    os.path.join(tgif_root_dir, 'split', 'trn_videoids.npy'),
    os.path.join(tgif_root_dir, 'split', 'val_videoids.npy'),
    os.path.join(tgif_root_dir, 'split', 'tst_videoids.npy'),
  ]
  tgif_ft_dir = os.path.join(tgif_root_dir, 'track_ft', ft_name)

  max_num_track = 10

  # num_tracks = []
  out_fts = []
  out_masks = []
  for vid_file in vid_files:
    vids = np.load(vid_file)
    for vid in vids:
      video = video_names[vid]
      ft_file = os.path.join(tgif_ft_dir, video + '.npy')
      if os.path.exists(ft_file):
        fts = np.load(ft_file)
        # num_tracks.append(fts.shape[0])
        num_ft = fts.shape[0]
        if num_ft == 0:
          fts = np.zeros((max_num_track, dim_ft), dtype=np.float32)
          mask = np.zeros((max_num_track,), dtype=np.float32)
        else:
          mask = np.ones((max_num_track,), dtype=np.float32)
          if num_ft > max_num_track:
            fts = fts[:max_num_track]
          elif num_ft < max_num_track:
            fts = np.concatenate([fts, np.zeros((max_num_track-num_ft,) + fts.shape[1:], dtype=np.float32)], 0)
            mask[num_ft:] = 0.
      else:
        fts = np.zeros((max_num_track, dim_ft), dtype=np.float32)
        mask = np.zeros((max_num_track,), dtype=np.float32)
      out_fts.append(fts)
      out_masks.append(mask)
  # print np.mean(num_tracks), np.median(num_tracks), np.percentile(num_tracks, 90)
  for vid in range(1915):
    ft_file = os.path.join(trecvid_root_dir, '16_track_ft', ft_name, '%d.npy'%vid)
    if os.path.exists(ft_file):
      fts = np.load(ft_file)
      num_ft = fts.shape[0]
      if num_ft == 0:
        fts = np.zeros((max_num_track, dim_ft), dtype=np.float32)
        mask = np.zeros((max_num_track,), dtype=np.float32)
      else:
        mask = np.ones((max_num_track,), dtype=np.float32)
        if num_ft > max_num_track:
          fts = fts[:max_num_track]
        elif num_ft < max_num_track:
          fts = np.concatenate([fts, np.zeros((max_num_track-num_ft,) + fts.shape[1:], dtype=np.float32)], 0)
          mask[num_ft:] = 0.
    else:
      fts = np.zeros((max_num_track, dim_ft), dtype=np.float32)
      mask = np.zeros((max_num_track,), dtype=np.float32)
    out_fts.append(fts)
    out_masks.append(mask)
  out_fts = np.array(out_fts, dtype=np.float32)
  out_masks = np.array(out_masks, dtype=np.float32)
  print out_fts.shape
  out_file = os.path.join(out_root_dir, 'sa_feature', out_ft_name, 'trn_ft.npz')
  np.savez_compressed(out_file, fts=out_fts, masks=out_masks)


def trecvid17_val_track_ft():
  root_dir = '/home/jiac/data/trecvid' # gpu8
  # vid_file = os.path.join(root_dir, '17', 'testing.2.subsets', 'tv17.vtt.url.list')
  # postfix = 'val_ft.2.npz'
  vid_file = os.path.join(root_dir, '17', 'testing.URLs.video.description.subtask')
  postfix = 'val_ft.npz'
  out_root_dir = '/home/jiac/data/trecvid2018'

  # ft_name = 'i3d_rgb'
  # out_ft_name = 'i3d'
  # dim_ft = 1024
  ft_name = 'resnet200'
  out_ft_name = 'resnet200'
  dim_ft = 2048
  max_num_track = 10

  vids = []
  with open(vid_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      vids.append(line[:pos])

  out_fts = []
  out_masks = []
  for vid in vids:
    ft_file = os.path.join(root_dir, '17_track_ft', vid + '.npy')
    if os.path.exists(ft_file):
      fts = np.load(ft_file)
      num_ft = fts.shape[0]
      if num_ft == 0:
        fts = np.zeros((max_num_track, dim_ft), dtype=np.float32)
        mask = np.zeros((max_num_track,), dtype=np.float32)
      else:
        mask = np.ones((max_num_track,), dtype=np.float32)
        if num_ft > max_num_track:
          fts = fts[:max_num_track]
        elif num_ft < max_num_track:
          fts = np.concatenate([fts, np.zeros((max_num_track-num_ft,) + fts.shape[1:], dtype=np.float32)], 0)
          mask[num_ft:] = 0.
    else:
      fts = np.zeros((max_num_track, dim_ft), dtype=np.float32)
      mask = np.zeros((max_num_track,), dtype=np.float32)
    out_fts.append(fts)
    out_masks.append(mask)
  out_fts = np.array(out_fts, dtype=np.float32)
  out_masks = np.array(out_masks, dtype=np.float32)
  print out_fts.shape
  out_file = os.path.join(out_root_dir, 'sa_feature', out_ft_name, postfix)
  np.savez_compressed(out_file, fts=out_fts, masks=out_masks)


if __name__ == '__main__':
  # merge_tgif_trecvid16_trn_track_ft()
  trecvid17_val_track_ft()

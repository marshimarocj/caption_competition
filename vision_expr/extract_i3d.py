import os
import json

import cv2
import imageio
import numpy as np

import i3d_wrapper.i3d_extract_graph


'''func
'''
def load_video(file):
  vid = cv2.VideoCapture(file)
  imgs = []
  while True:
    ret, img = vid.read()
    if not ret:
      break
    imgs.append(img)
  return imgs


'''expr
'''
def extract_vtt():
  root_dir = '/mnt/data2/jiac/vtt_raw' # neptune
  lst_file = os.path.join(root_dir, '18.lst')
  video_dir = os.path.join(root_dir, '18')
  track_dir = os.path.join(root_dir, '18_kcf_track')
  model_file = '/home/jiac/toolkit/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
  out_dir = os.path.join(root_dir, '18_track_ft', 'i3d_rgb')

  graph = i3d_wrapper.i3d_extract_graph.I3dExtractGraph('RGB')
  graph.load_model(model_file)
  print 'load complete'

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      names.append(name)

  for name in names:
    video_file = os.path.join(video_dir, name + '.mp4')
    track_file = os.path.join(track_dir, name + '.json')
    if not os.path.exists(track_file):
      continue

    imgs = load_video(video_file)
    img_h, img_w, _ = imgs[0].shape

    with open(track_file) as f:
      paths = json.load(f)
    out_fts = []
    for path in paths:
      max_w = max([d['w'] for d in path])
      max_h = max([d['h'] for d in path])
      crop_size = max(max_w, max_h)
      crop_imgs = []
      for d in path:
        frame = d['frame']
        if frame >= len(imgs):
          break

        cx = d['x'] + d['w']/2
        cy = d['y'] + d['h']/2
        crop_img = 128*np.ones((crop_size, crop_size, 3), dtype=np.uint8)
        min_x = cx - crop_size/2
        min_y = cy - crop_size/2
        max_x = min_x + crop_size
        max_y = min_y + crop_size
        crop_min_x = max(-min_x, 0)
        crop_max_x = crop_size - max(max_x - img_w, 0)
        crop_min_y = max(-min_y, 0)
        crop_max_y = crop_size - max(max_y - img_h, 0)
        # print min_x, max_x, min_y, max_y
        # print crop_min_x, crop_max_x, crop_min_y, crop_max_y
        crop_img[crop_min_y:crop_max_y, crop_min_x:crop_max_x] = imgs[frame][max(min_y, 0):max_y, max(min_x, 0):max_x]

        crop_img = cv2.resize(crop_img, (224, 224))
        crop_img = crop_img[:, :, ::-1]
        crop_img = crop_img / 255.
        crop_img = crop_img * 2 - 1.
        crop_imgs.append(crop_img)
      i = 0
      valid_len = len(crop_imgs)
      while len(crop_imgs) < 64:
        crop_imgs.append(crop_imgs[i%valid_len])
        i += 1
      fts = []
      for i in range(0, len(crop_imgs), 16):
        if i + 64 > len(crop_imgs):
          break
        ft = graph.extract_feature([crop_imgs[i:i+64]])
        fts.append(ft[0])
      fts = np.array(fts)
      # print fts.shape
      for i in range(4)[::-1]:
        fts = np.mean(fts, i)
      out_fts.append(fts)
    out_file = os.path.join(out_dir, name + '.npy')
    np.save(out_file, out_fts)


def extract_tgif():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'split.0.lst')
  # root_dir = '/home/jiac/data/tgif' # gpu8
  # lst_file = os.path.join(root_dir, 'split.1.lst')
  gif_dir = os.path.join(root_dir, 'gif')
  track_dir = os.path.join(root_dir, 'kcf_track')
  model_file = '/home/jiac/models/tf/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
  out_dir = os.path.join(root_dir, 'track_ft', 'i3d_rgb')

  graph = i3d_wrapper.i3d_extract_graph.I3dExtractGraph('RGB')
  graph.load_model(model_file)
  print 'load complete'

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      names.append(name)

  for name in names:
    gif_file = os.path.join(gif_dir, name + '.gif')
    track_file = os.path.join(track_dir, name + '.json')
    out_file = os.path.join(out_dir, name + '.npy')
    if not os.path.exists(track_file):
      continue
    if os.path.exists(out_file):
      continue

    try:
      gif = imageio.mimread(gif_file, memtest=False)
    except:
      continue
    if len(gif[0].shape) < 3:
      continue
    imgs = []
    for i in range(len(gif)):
      img = np.asarray(gif[i][:, :, :3], dtype=np.uint8)
      imgs.append(img[:, :, ::-1].copy())
    img_h, img_w, _ = imgs[0].shape

    with open(track_file) as f:
      paths = json.load(f)
    out_fts = []
    for path in paths:
      max_w = max([d['w'] for d in path])
      max_h = max([d['h'] for d in path])
      crop_size = max(max_w, max_h)
      crop_imgs = []
      for d in path:
        frame = d['frame']
        if frame >= len(imgs):
          break

        cx = d['x'] + d['w']/2
        cy = d['y'] + d['h']/2
        crop_img = 128*np.ones((crop_size, crop_size, 3), dtype=np.uint8)
        min_x = cx - crop_size/2
        min_y = cy - crop_size/2
        max_x = min_x + crop_size
        max_y = min_y + crop_size
        crop_min_x = max(-min_x, 0)
        crop_max_x = crop_size - max(max_x - img_w, 0)
        crop_min_y = max(-min_y, 0)
        crop_max_y = crop_size - max(max_y - img_h, 0)
        # print min_x, max_x, min_y, max_y
        # print crop_min_x, crop_max_x, crop_min_y, crop_max_y
        crop_img[crop_min_y:crop_max_y, crop_min_x:crop_max_x] = imgs[frame][max(min_y, 0):max_y, max(min_x, 0):max_x]

        crop_img = cv2.resize(crop_img, (224, 224))
        crop_img = crop_img[:, :, ::-1]
        crop_img = crop_img / 255.
        crop_img = crop_img * 2 - 1.
        crop_imgs.append(crop_img)
      i = 0
      valid_len = len(crop_imgs)
      while len(crop_imgs) < 64:
        crop_imgs.append(crop_imgs[i%valid_len])
        i += 1
      fts = []
      for i in range(0, len(crop_imgs), 16):
        if i + 64 > len(crop_imgs):
          break
        ft = graph.extract_feature([crop_imgs[i:i+64]])
        fts.append(ft[0])
      fts = np.array(fts)
      # print fts.shape
      for i in range(4)[::-1]:
        fts = np.mean(fts, i)
      out_fts.append(fts)
    out_file = os.path.join(out_dir, name + '.npy')
    np.save(out_file, out_fts)


def extract_missing_tgif():
  # root_dir = '/home/jiac/data2/tgif' # gpu9
  # data_root_dir = os.path.join(root_dir, 'TGIF-Release', 'data')
  # lst_file = os.path.join(data_root_dir, 'split.0.lst')
  root_dir = '/home/jiac/data/tgif' # gpu8
  data_root_dir = root_dir
  valid_video_lst_file = os.path.join(root_dir, 'aux', 'int2video.npy')
  lst_file = os.path.join(data_root_dir, 'split.3.lst')
  video_dir = os.path.join(data_root_dir, 'mp4')
  track_dir = os.path.join(data_root_dir, 'kcf_track')
  model_file = '/home/jiac/models/tf/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
  out_dir = os.path.join(data_root_dir, 'track_ft', 'i3d_rgb')

  valid_videos = np.load(valid_video_lst_file)
  valid_videos = set(valid_videos.tolist())

  graph = i3d_wrapper.i3d_extract_graph.I3dExtractGraph('RGB')
  graph.load_model(model_file)
  print 'load complete'

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      if name in valid_videos and num_frame > 0:
        names.append(name)

  for name in names:
    video_file = os.path.join(video_dir, name + '.mp4')
    track_file = os.path.join(track_dir, name + '.json')
    out_file = os.path.join(out_dir, name + '.npy')
    if os.path.exists(out_file):
      continue

    print name

    imgs = load_video(video_file)
    img_h, img_w, _ = imgs[0].shape

    with open(track_file) as f:
      paths = json.load(f)
    out_fts = []
    for path in paths:
      max_w = max([d['w'] for d in path])
      max_h = max([d['h'] for d in path])
      crop_size = max(max_w, max_h)
      crop_imgs = []
      for d in path:
        frame = d['frame']
        if frame >= len(imgs):
          break

        cx = d['x'] + d['w']/2
        cy = d['y'] + d['h']/2
        crop_img = 128*np.ones((crop_size, crop_size, 3), dtype=np.uint8)
        min_x = cx - crop_size/2
        min_y = cy - crop_size/2
        max_x = min_x + crop_size
        max_y = min_y + crop_size
        crop_min_x = max(-min_x, 0)
        crop_max_x = crop_size - max(max_x - img_w, 0)
        crop_min_y = max(-min_y, 0)
        crop_max_y = crop_size - max(max_y - img_h, 0)
        crop_img[crop_min_y:crop_max_y, crop_min_x:crop_max_x] = imgs[frame][max(min_y, 0):max_y, max(min_x, 0):max_x]

        crop_img = cv2.resize(crop_img, (224, 224))
        crop_img = crop_img[:, :, ::-1]
        crop_img = crop_img / 255.
        crop_img = crop_img * 2 - 1.
        crop_imgs.append(crop_img)
      i = 0
      valid_len = len(crop_imgs)
      while len(crop_imgs) < 64:
        crop_imgs.append(crop_imgs[i%valid_len])
        i += 1
      fts = []
      for i in range(0, len(crop_imgs), 16):
        if i + 64 > len(crop_imgs):
          break
        ft = graph.extract_feature([crop_imgs[i:i+64]])
        fts.append(ft[0])
      fts = np.array(fts)
      for i in range(4)[::-1]:
        fts = np.mean(fts, i)
      out_fts.append(fts)
    np.save(out_file, out_fts)


if __name__ == '__main__':
  # extract_vtt()
  # extract_tgif()
  extract_missing_tgif()

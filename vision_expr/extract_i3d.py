import os
import json

import cv2
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
  lst_file = os.path.join(root_dir, '16.lst')
  video_dir = os.path.join(root_dir, '16')
  track_dir = os.path.join(root_dir, '16_kcf_track')
  model_file = '/home/jiac/toolkit/kinetics-i3d/data/checkpoints/rgb_imagenet/model.ckpt'
  out_dir = os.path.join(root_dir, '16_track_ft', 'i3d_rgb')

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
        print min_x, max_x, min_y, max_y
        print crop_min_x, crop_max_x, crop_min_y, crop_max_y
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
      ft = np.mean(np.mean(np.mean(fts, 0), 1), 2)
      out_fts.append(ft)
    out_file = os.path.join(out_dir, name + '.npy')
    np.save(out_file, out_fts)


if __name__ == '__main__':
  extract_vtt()

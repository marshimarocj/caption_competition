import os
import argparse
import subprocess

import numpy as np
import imageio
from scipy.io import loadmat
import cv2


colormap = [ # bgr
  [228,26,28][::-1],
  [55,126,184][::-1],
  [77,175,74][::-1],
  [152,78,163][::-1],
  [255,127,0][::-1],
]


'''func
'''


'''expr
'''
def prepare_lst_for_matlab():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  video_dir = os.path.join(root_dir, 'mp4')
  detect_dir = os.path.join(root_dir, 'obj_detect')
  out_file = os.path.join(root_dir, 'split.0.lst')

  chunk = 0
  split = 4

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      url = line[:pos]
      pos = url.rfind('/')
      name = url[pos+1:]
      name, _ = os.path.splitext(name)
      names.append(name)
  split_gap = (len(names) + split - 1) / split

  with open(out_file, 'w') as fout:
    for name in names[chunk*split_gap : (chunk+1)*split_gap]:
      detect_file = os.path.join(detect_dir, name + '.npz')
      if not os.path.exists(detect_file):
        continue

      video_file = os.path.join(video_dir, name + '.mp4')
      vid = cv2.VideoCapture(video_file)
      num_frame = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
      if num_frame < 5:
        print name, num_frame
        continue

      data = np.load(detect_file)
      if 'scores' not in data:
        continue
      scores = data['scores']
      num = scores.shape[0]
      num = (num + 2) / 3
      fout.write('%s %d\n'%(name, num))


def viz_tracking():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  track_root_dir = os.path.join(root_dir, 'track')
  gif_dir = os.path.join(root_dir, 'gif')
  viz_dir = os.path.join(root_dir, 'viz')

  gap = 16

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      url = line[:pos]
      pos = url.rfind('/')
      name = url[pos+1:]
      name, _ = os.path.splitext(name)
      names.append(name)

  for name in names[:100]:
    gif_file = os.path.join(gif_dir, name + '.gif')
    if not os.path.exists(gif_file):
      continue
    gif = imageio.mimread(gif_file, memtest=False)
    if len(gif[0].shape) < 3:
      continue

    track_dir = os.path.join(track_root_dir, name)
    num = len(os.listdir(track_dir))
    frame = 0
    out_imgs = []
    for i in range(num):
      track_file = os.path.join(track_dir, '%d.mat'%(i*gap))
      data = loadmat(track_file)
      bboxs = data['results']
      scores = data['scores']
      num_rect, num_frame = scores.shape
      for i in range(num_frame):
        if frame >= len(gif):
          break
        img = np.asarray(gif[frame][:, :, :3], dtype=np.uint8) # rgb
        # print type(img), img.shape, img.dtype
        canvas = img[:, :, ::-1].copy()
        for j in range(num_rect):
          x, y, w, h = bboxs[j, i]
          x = int(x)
          y = int(y)
          w = int(w)
          h = int(h)
          new_canvas = canvas.copy()
          cv2.rectangle(new_canvas, (x, y), (x+w, y+h), colormap[j%len(colormap)], 2);
          # print colormap[j%len(colormap)]
          score = scores[j, i]
          canvas = canvas * (1. - score) + score * new_canvas
          canvas = canvas.astype(np.uint8)
        canvas = canvas[:, :, ::-1]
        canvas = canvas.astype(np.uint8)
        out_imgs.append(canvas)
        frame += 1

    out_file = os.path.join(viz_dir, name + '.gif')
    imageio.mimsave(out_file, out_imgs)


def kcf_tracking():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'split.0.lst')
  obj_detect_root_dir = os.path.join(root_dir, 'obj_detect')
  video_dir = os.path.join(root_dir, 'mp4')
  track_root_dir = os.path.join(root_dir, 'kcf_track')

  gap = 16

  name_nums = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num = int(data[1])
      name_nums.append((name, num))

  for name, num in name_nums[:100]:
    video_file = os.path.join(video_dir, name + '.mp4')
    bbox_dir = os.path.join(obj_detect_root_dir, name)
    track_dir = os.path.join(track_root_dir, name)
    if not os.path.exists(track_dir):
      os.mkdir(track_dir)
    print name

    cmd = [
      '/home/jiac/code/cpp/kcf/tracker',
      video_file,
      bbox_dir + '/',
      track_dir + '/',
      str(num), str(gap),
    ]
    p = subprocess.Popen(cmd)
    p.wait()


def viz_kcf_tracking():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  track_root_dir = os.path.join(root_dir, 'kcf_track')
  gif_dir = os.path.join(root_dir, 'gif')
  viz_dir = os.path.join(root_dir, 'kcf_viz')

  gap = 16

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      url = line[:pos]
      pos = url.rfind('/')
      name = url[pos+1:]
      name, _ = os.path.splitext(name)
      names.append(name)

  for name in names[:100]:
    gif_file = os.path.join(gif_dir, name + '.gif')
    if not os.path.exists(gif_file):
      continue
    gif = imageio.mimread(gif_file, memtest=False)
    if len(gif[0].shape) < 3:
      continue

    track_dir = os.path.join(track_root_dir, name)
    num = len(os.listdir(track_dir))
    frame = 0
    out_imgs = []
    for i in range(num):
      track_file = os.path.join(track_dir, '%d.track'%(i*gap))
      all_bboxs = []
      all_scores = []
      with open(track_file) as f:
        for line in f:
          line = line.strip()
          data = line.split(' ')
          bboxs = []
          scores = []
          for i in range(0, len(data), 5):
            bbox = [int(d) for d in data[i:i+4]]
            score = float(data[i+4])
            bboxs.append(bbox)
            scores.append(score)
          all_bboxs.append(bboxs)
          all_scores.append(scores)
      num_rect = len(all_scores)
      num_frame = len(all_scores[0])
      for i in range(num_frame):
        if frame >= len(gif):
          break
        img = np.asarray(gif[frame][:, :, :3], dtype=np.uint8) # rgb
        canvas = img[:, :, ::-1].copy()
        for j in range(num_rect):
          x, y, w, h = all_boxes[j][i]
          new_canvas = canvas.copy()
          cv2.rectangle(new_canvas, (x, y), (x+w, y+h), colormap[j%len(colormap)], 2);
          score = scores[j][i]
          canvas = canvas * (1. - score) + score * new_canvas
          canvas = canvas.astype(np.uint8)
        canvas = canvas[:, :, ::-1] # rgb
        canvas = canvas.astype(np.uint8)
        out_imgs.append(canvas)

    out_file = os.path.join(viz_dir, name + '.gif')
    imageio.mimsave(out_file, out_imgs)


if __name__ == '__main__':
  # prepare_lst_for_matlab()
  # viz_tracking()
  # kcf_tracking()
  viz_kcf_tracking()

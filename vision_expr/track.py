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
def load_track(file, reverse=False):
  with open(file) as f:
    all_boxs = []
    all_scores = []
    for line in f:
      line = line.strip()
      data = line.split(' ')
      num = len(data) / 5 * 5
      boxs = []
      scores = []
      for i in range(0, num, 5):
        x, y, w, h = [int(d) for d in data[i:i+4]]
        score = float(data[i+4])
        boxs.append((x, y, w, h))
        scores.append(score)
      if reverse:
        boxs = boxs[::-1]
        scores = scores[::-1]
      all_boxs.append(boxs)
      all_scores.append(scores)
    all_boxs = np.array(all_boxs, dtype=np.float32)
    all_scores = np.array(all_scores, dtype=np.float32)
  return all_boxs, all_scores


def bbox_intersect(lboxs, rboxs):
  lboxs = np.expand_dims(lboxs, 1)
  rboxs = np.expand_dims(rboxs, 0)
  x1 = np.maximum(lboxs[:, :, 0], rboxs[:, :, 0])
  y1 = np.maximum(lboxs[:, :, 1], rboxs[:, :, 1])
  x2 = np.minimum(lboxs[:, :, 0] + lboxs[:, :, 2], rboxs[:, :, 0] + rboxs[:, :, 2])
  y2 = np.minimum(lboxs[:, :, 1] + lboxs[:, :, 3], rboxs[:, :, 1] + rboxs[:, :, 2])
  w = np.maximum(x2 - x1, np.zeros(x1.shape))
  h = np.maximum(y2 - y1, np.zeros(y1.shape))
  return w * h


def bbox_union(lboxs, rboxs):
  intersect = bbox_intersect(lboxs, rboxs)
  lareas = lboxs[:, 2] * lboxs[:, 3]
  rareas = rboxs[:, 2] * rboxs[:, 3]
  union = np.expand_dims(lareas, 1) + np.expand_dims(rareas, 0) - intersect
  return union


'''expr
'''
def prepare_num_frame_lst():
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

      data = np.load(detect_file)
      if 'scores' not in data:
        continue
      fout.write('%s %d\n'%(name, num_frame))


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

  gap = 8

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
      str(num), str(gap), '0',
    ]
    p = subprocess.Popen(cmd)
    p.wait()

    cmd[-1] = '1'
    p = subprocess.Popen(cmd)
    p.wait()


def viz_kcf_tracking():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  track_root_dir = os.path.join(root_dir, 'kcf_track')
  gif_dir = os.path.join(root_dir, 'gif')
  viz_dir = os.path.join(root_dir, 'kcf_viz')

  gap = 8
  score_threshold = 0.2

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
    num = (len(os.listdir(track_dir)) + 1)/2
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
      num_frame = len(all_scores[0]) if num_rect > 0 else 0
      for i in range(num_frame):
        if frame >= len(gif):
          break
        img = np.asarray(gif[frame][:, :, :3], dtype=np.uint8) # rgb
        canvas = img[:, :, ::-1].copy()
        for j in range(num_rect):
          x, y, w, h = all_bboxs[j][i]
          score = all_scores[j][i]
          if score >= score_threshold:
            cv2.rectangle(canvas, (x, y), (x+w, y+h), colormap[j%len(colormap)], 2);
          # new_canvas = canvas.copy()
          # cv2.rectangle(new_canvas, (x, y), (x+w, y+h), colormap[j%len(colormap)], 2);
          # canvas = canvas * (1. - score) + score * new_canvas
          # canvas = canvas.astype(np.uint8)
        canvas = canvas[:, :, ::-1] # rgb
        canvas = canvas.astype(np.uint8)
        out_imgs.append(canvas)
        frame += 1

    out_file = os.path.join(viz_dir, name + '.gif')
    imageio.mimsave(out_file, out_imgs)


def associate_forward_backward():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'split.0.lst')
  track_root_dir = os.path.join(root_dir, 'kcf_track')

  gap = 8
  score_threshold = 0.2

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

  iou_threshold = 0.3

  # ious = []
  for name, num_frame in name_frames[:100]:
    track_dir = os.path.join(track_root_dir, name)
    for f in range(0, num_frame, gap):
      forward_file = os.path.join(track_dir, '%d.track'%f)
      backward_file = os.path.join(track_dir, '%d.rtrack'%f)
      if not os.path.exists(backward_file):
        continue

       # (num_obj, num_frame, 4), (num_obj, num_frame)
      forward_boxs, forward_scores = load_track(forward_file)
      backward_boxs, backward_scores = load_track(backward_file, reverse=True)
      num_forward = forward_boxs.shape[0]
      num_backward = backward_boxs.shape[0]
      if num_forward == 0 or num_backward == 0:
        continue

      forward_valid = forward_scores >= score_threshold
      forward_valid = np.repeat(np.expand_dims(forward_valid, 2), 4, 2).astype(np.bool_)
      backward_valid = backward_scores >= score_threshold
      backward_valid = np.repeat(np.expand_dims(backward_valid, 2), 4, 2).astype(np.bool_)
      forward_boxs = np.where(forward_valid, forward_boxs, np.zeros(forward_boxs.shape))
      backward_boxs = np.where(backward_valid, backward_boxs, np.zeros(backward_boxs.shape))

      intersect_volumes = np.zeros((num_forward, num_backward))
      union_volumes = np.zeros((num_forward, num_backward))
      for i in range(gap):
        intersect = bbox_intersect(forward_boxs[:, i], backward_boxs[:, i]) # (num_forward, num_backward)
        intersect_volumes += intersect
        union = bbox_union(forward_boxs[:, i], backward_boxs[:, i])
        union_volumes += union
      ious = intersect_volumes / union_volumes
  #     ious += np.max(iou, 0).tolist()
  #     ious += np.max(iou, 1).tolist()
  # print np.median(ious), np.mean(ious), np.percentile(ious, 10), np.percentile(ious, 90)

      pairs = [] # greedy
      for i in range(min(num_forward, num_backward)):
        idx = np.unravel_index(np.argmax(ious, axis=None), ious.shape)
        if ious[idx] < iou_threshold:
          break
        pairs.append((idx[0], idx[1], ious[idx]))
        ious[idx[0]] = 0.
        ious[:, idx[1]] = 0.
      out_file = os.path.join(track_dir, '%d.associate'%f)
      with open(out_file, 'w') as fout:
        for r, c, iou in pairs:
          fout.write('%d %d %f\n'%(r, c, iou))


def generate_tracklet():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'split.0.lst')
  track_root_dir = os.path.join(root_dir, 'kcf_track')

  gap = 8

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

  for name, num_frame in name_frames[1:2]:
    track_dir = os.path.join(track_root_dir, name)
    associates = []
    for frame in range(0, num_frame, gap):
      associate_file = os.path.join(track_dir, '%d.associate'%frame)
      if not os.path.exists(associate_file):
        associates.append({})
        continue

      associate = {}
      with open(associate_file) as f:
        for line in f:
          line = line.strip()
          data = line.split(' ')
          associate[int(data[0])] = {'bid': int(data[1])}

      forward_file = os.path.join(track_dir, '%d.track'%frame)
      backward_file = os.path.join(track_dir, '%d.rtrack'%frame)
      forward_boxs, forward_scores = load_track(forward_file)
      backward_boxs, backward_scores = load_track(backward_file, True)

      for fid in associate:
        bid = associate[fid]['bid']
        alpha = np.arange(gap) / (gap-1)
        alpha = np.expand_dims(alpha, 1)
        boxes = forward_boxs[fid] * (1. - alpha) + backward_boxs[bid] * alpha
        associate[fid]['boxs'] = boxes

      associates.append(associate)
    tracklets = []
    buffers = []
    for associate in associates:
      bid2buffer = {}
      for d in buffers:
        bid2buffer[d['bid']]  = d['boxs']
      print bid2buffer.keys()
      buffers = []
      for fid in associate:
        # print fid, fid in bid2buffer
        boxs = associate[fid]['boxs']
        bid = associate[fid]['bid']
        if fid in bid2buffer:
          boxs = np.concatenate([bid2buffer[fid], boxs], 1)
          buffers.append({'bid': bid, 'boxs': boxs})
        else:
          tracklets.append(boxs)
    for d in buffers:
      tracklets.append(d['boxs'])
    print name, num_frame, len(tracklets)


if __name__ == '__main__':
  # prepare_num_frame_lst()
  # viz_tracking()
  # kcf_tracking()
  # viz_kcf_tracking()
  # associate_forward_backward()
  generate_tracklet()

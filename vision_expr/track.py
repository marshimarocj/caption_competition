import os
import argparse
import subprocess

import numpy as np
# import imageio
# from scipy.io import loadmat
import cv2


colormap = [ # bgr
  [228,26,28][::-1],
  [55,126,184][::-1],
  [77,175,74][::-1],
  [152,78,163][::-1],
  [255,127,0][::-1],
]

colormap12 = [
  [166,206,227][::-1],
  [31,120,180][::-1],
  [178,223,138][::-1],
  [51,160,44][::-1],
  [251,154,153][::-1],
  [227,26,28][::-1],
  [253,191,111][::-1],
  [255,127,0][::-1],
  [202,178,214][::-1],
  [106,61,154][::-1],
  [255,255,153][::-1],
  [177,89,40][::-1],
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
  y2 = np.minimum(lboxs[:, :, 1] + lboxs[:, :, 3], rboxs[:, :, 1] + rboxs[:, :, 3])
  w = np.maximum(x2 - x1, np.zeros(x1.shape))
  h = np.maximum(y2 - y1, np.zeros(y1.shape))
  return w * h


def bbox_union(lboxs, rboxs):
  intersect = bbox_intersect(lboxs, rboxs)
  lareas = lboxs[:, 2] * lboxs[:, 3]
  rareas = rboxs[:, 2] * rboxs[:, 3]
  union = np.expand_dims(lareas, 1) + np.expand_dims(rareas, 0) - intersect
  return union


# max sum
def viterbi_decoding(edges):
  num_step = len(edges) + 1

  forward_sums = [np.zeros(edges[0].shape[0],)]
  prevs = [-np.ones((edges[0].shape[0],), dtype=np.int32)]
  backward_sums = [np.zeros((edges[-1].shape[1],))]
  nexts = [-np.ones((edges[-1].shape[1],), dtype=np.int32)]

  for i in range(0, num_step-1):
    w = np.where(edges[i] > 0, edges[i] + np.expand_dims(forward_sums[i], 1), edges[i])
    if w.size > 0:
      forward_sums.append(np.max(w, 0))
      prevs.append(np.argmax(w, 0))
    else:
      forward_sums.append(np.zeros((w.shape[1],)))
      prevs.append(-np.ones((w.shape[1],)))
  for i in range(1, num_step):
    w = np.where(edges[-i] > 0, edges[-i] + np.expand_dims(backward_sums[i-1], 0), edges[-i])
    if w.size > 0:
      backward_sums.append(np.max(w, 1))
      nexts.append(np.argmax(w, 1))
    else:
      backward_sums.append(np.zeros((w.shape[0],)))
      nexts.append(-np.ones((w.shape[0],)))

  total_sums = []
  max_sum = 0.
  max_id = -1
  max_step = -1
  for i in range(num_step):
    total_sum = forward_sums[i] + backward_sums[-i-1]
    total_sums.append(total_sum)
    if total_sum.size > 0 and np.max(total_sum) > max_sum:
      max_sum = np.max(total_sum)
      max_id = np.argmax(total_sum)
      max_step = i
  if max_sum == 0:
    return max_sum, []

  t = max_step
  i = max_id
  path = []
  while t >= 0:
    path.append((t, i))
    if forward_sums[t][i] == 0:
      break
    i = prevs[t][i]
    t -= 1
  path = path[::-1]

  t = max_step
  i = max_id
  # print (t, i)
  while t < num_step-1:
    if backward_sums[-t-1][i] == 0:
      break
    i = nexts[-t-1][i]
    t += 1
    path.append((t, i))

  return max_sum, path


def remove_path_node_from_graph(edges, path):
  num_step = len(edges) + 1
  for t, i in path:
    if t == 0:
      edges[t][i] = 0
    elif t == num_step-1:
      edges[t-1][:, i] = 0
    else:
      edges[t][i] = 0
      edges[t-1][:, i] = 0


def calc_boxs_for_path(path, all_forward_boxs, all_backward_boxs, alphas):
  num_step = len(path)
  all_boxs = []
  for i in range(num_step-1):
    step = path[i][0]
    fid = path[i][1]
    bid = path[i+1][1]
    boxs = all_forward_boxs[step][fid] * (1. - alphas) + all_backward_boxs[step][bid] * alphas
    all_boxs.append(boxs)
  return all_boxs


def calc_area_maxmin_ratio(all_boxs):
  min_area = 1e10
  max_area = 0.
  for boxs in all_boxs:
    areas = boxs[:, 2] * boxs[:, 3]
    if np.max(areas) > max_area:
      max_area = np.max(areas)
    if np.min(areas) < min_area:
      min_area = np.min(areas)
  return max_area / min_area


def cut_balance_maxmin_ratio(path, all_boxs, begin, end, max_num_step, maxmin_ratio_threshold, refined_paths):
  maxmin_ratio = calc_area_maxmin_ratio(all_boxs[begin:end])
  if maxmin_ratio < maxmin_ratio_threshold:
    refined_paths.append(path[begin:end+1])
    return
  if end-begin <= max_num_step:
    return

  lmin = []
  lmax = []
  rmin = []
  rmax = []
  for i in range(begin, end):
    if i == begin:
      area = all_boxs[i][0, 2] * all_boxs[i][0, 3]
      lmin.append(area)
      lmax.append(area)
    else:
      areas = all_boxs[i-1][:, 2] * all_boxs[i-1][:, 3]
      lmin.append(min(lmin[-1], np.min(areas)))
      lmax.append(max(lmax[-1], np.max(areas)))
  for i in range(end-1, begin-1, -1):
    if i == end-1:
      area = all_boxs[i][-1, 2] * all_boxs[i][-1, 3]
      rmin.append(area)
      rmax.append(area)
    else:
      areas = all_boxs[i][:, 2] * all_boxs[i][:, 3]
      rmin.append(min(rmin[-1], np.min(areas)))
      rmax.append(max(rmax[-1], np.max(areas)))
  rmin = rmin[::-1]
  rmax = rmax[::-1]
  min_ratio = 1e10
  cut = -1
  for i in range(begin, end):
    lratio = lmax[i-begin] / lmin[i-begin]
    rratio = rmax[i-begin] / rmin[i-begin]
    if min(lratio, rratio) < min_ratio:
      min_ratio = min(lratio, rratio)
      cut = i
  if cut > begin:
    cut_balance_maxmin_ratio(path, all_boxs, begin, cut, max_num_step, maxmin_ratio_threshold, refined_paths)
  if cut < end:
    cut_balance_maxmin_ratio(path, all_boxs, cut+1, end, max_num_step, maxmin_ratio_threshold, refined_paths)


def load_path(path_file):
  paths = []
  with open(path_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      path = []
      for d in data:
        fields = d.split(',')
        path.append((int(fields[0]), int(fields[1])))
      paths.append(path)

  return paths


'''expr
'''
def prepare_num_frame_lst():
  # root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  root_dir = '/home/jiac/data/tgif' # gpu8
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  video_dir = os.path.join(root_dir, 'mp4')
  detect_dir = os.path.join(root_dir, 'obj_detect')
  chunk = 1
  out_file = os.path.join(root_dir, 'split.%d.lst'%chunk)

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


def prepare_num_frame_lst_vtt():
  root_dir = '/mnt/data2/jiac/vtt_raw' # neptune
  video_dir = os.path.join(root_dir, '18')
  detect_dir = os.path.join(root_dir, '18_obj_detect')
  out_file = os.path.join(root_dir, '18.lst')

  names = os.listdir(video_dir)
  with open(out_file, 'w') as fout:
    for name in names:
      name, _ = os.path.splitext(name)
      video_file = os.path.join(video_dir, name + '.mp4')
      vid = cv2.VideoCapture(video_file)
      num_frame = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

      detect_file = os.path.join(detect_dir, name + '.npz')
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
  # root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  # lst_file = os.path.join(root_dir, 'split.0.lst')
  root_dir = '/home/jiac/data/tgif/' # gpu8
  lst_file = os.path.join(root_dir, 'split.1.lst')
  obj_detect_root_dir = os.path.join(root_dir, 'obj_detect')
  video_dir = os.path.join(root_dir, 'mp4')
  track_root_dir = os.path.join(root_dir, 'kcf_track')

  # root_dir = '/mnt/data2/jiac/vtt_raw' # neptune
  # lst_file = os.path.join(root_dir, '18.lst')
  # obj_detect_root_dir = os.path.join(root_dir, '18_obj_detect')
  # video_dir = os.path.join(root_dir, '18')
  # track_root_dir = os.path.join(root_dir, '18_kcf_track')

  gap = 8
  num_thread = 4

  name_nums = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num = int(data[1])
      name_nums.append((name, num))

  # for name, num in name_nums[:100]:
  ps = []
  for name, num in name_nums:
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
    # p.wait()
    ps.append(p)

    cmd[-1] = '1'
    p = subprocess.Popen(cmd)
    # p.wait()
    ps.append(p)

    if len(ps) >= num_thread:
      for p in ps:
        p.wait()
      ps = []


def viz_kcf_tracking():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  track_root_dir = os.path.join(root_dir, 'kcf_track')
  gif_dir = os.path.join(root_dir, 'gif')
  viz_dir = os.path.join(root_dir, 'kcf_viz')

  gap = 8
  score_threshold = 0.2
  reverse = True

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

  for name in names[10:100]:
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
      if reverse:
        track_file = os.path.join(track_dir, '%d.rtrack'%(i*gap))
      else:
        track_file = os.path.join(track_dir, '%d.track'%(i*gap))
      if not os.path.exists(track_file):
        break
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
          if reverse:
            bboxs = bboxs[::-1]
            scores = scores[::-1]
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
        canvas = canvas[:, :, ::-1] # rgb
        canvas = canvas.astype(np.uint8)
        out_imgs.append(canvas)
        frame += 1

    if not reverse:
      out_file = os.path.join(viz_dir, name + '.gif')
    else:
      out_file = os.path.join(viz_dir, name + '.reverse.gif')
    imageio.mimsave(out_file, out_imgs)


def associate_forward_backward():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'split.0.lst')
  track_root_dir = os.path.join(root_dir, 'kcf_track')

  gap = 8
  iou_threshold = 0.5
  # gap = 4
  # iou_threshold = 0.5

  # score_threshold = 0.2

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

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

      intersect_volumes = np.zeros((num_forward, num_backward))
      union_volumes = np.zeros((num_forward, num_backward))
      for i in range(gap):
        intersect = bbox_intersect(forward_boxs[:, i], backward_boxs[:, i]) # (num_forward, num_backward)
        intersect_volumes += intersect
        union = bbox_union(forward_boxs[:, i], backward_boxs[:, i])
        union_volumes += union
      ious = intersect_volumes / union_volumes

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


def viz_association():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'split.0.lst')
  gif_dir = os.path.join(root_dir, 'gif')
  track_root_dir = os.path.join(root_dir, 'kcf_track')
  viz_dir = os.path.join(root_dir, 'kcf_viz')

  gap = 8

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

  for name, num_frame in name_frames[:100]:
    gif_file = os.path.join(gif_dir, name + '.gif')
    if not os.path.exists(gif_file):
      continue
    gif = imageio.mimread(gif_file, memtest=False)
    if len(gif[0].shape) < 3:
      continue
    imgs = []
    for i in range(len(gif)):
      img = np.asarray(gif[i][:, :, :3], dtype=np.uint8)
      imgs.append(img[:, :, ::-1].copy())

    track_dir = os.path.join(track_root_dir, name)
    for frame in range(0, num_frame, gap):
      associate_file = os.path.join(track_dir, '%d.associate'%frame)
      if not os.path.exists(associate_file):
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

      cnt = 0
      for fid in associate:
        bid = associate[fid]['bid']
        alphas = np.arange(gap) / float(gap-1)
        alphas = np.expand_dims(alphas, 1)
        boxes = forward_boxs[fid] * (1. - alphas) + backward_boxs[bid] * alphas
        for i in range(gap):
          f = frame + i
          img = imgs[f]
          x, y, w, h = [int(d) for d in boxes[i]]
          cv2.rectangle(img, (x, y), (x+w, y+h), colormap12[cnt%len(colormap12)], 2);
        cnt += 1
    out_imgs = []
    for img in imgs:
      out_imgs.append(img[:, :, ::-1])
    out_file = os.path.join(viz_dir, name + '.associate.gif')
    imageio.mimsave(out_file, out_imgs)


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

  for name, num_frame in name_frames[:100]:
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
        alphas = np.arange(gap) / float(gap-1)
        alphas = np.expand_dims(alphas, 1)
        boxes = forward_boxs[fid] * (1. - alphas) + backward_boxs[bid] * alphas
        # boxes = []
        # for alpha in alphas:
        #   boxes.append(forward_boxs[fid, 0] * (1. - alpha) + backward_boxs[bid, -1] * alpha)
        # boxes = np.array(boxes)
        associate[fid]['boxs'] = boxes
      associates.append(associate)

    tracklets = []
    buffers = []
    for f, associate in enumerate(associates):
      bid2buffer = {}
      for d in buffers:
        bid2buffer[d['bid']]  = (d['start'], d['boxs'])
      buffers = []
      for fid in associate:
        boxs = associate[fid]['boxs']
        bid = associate[fid]['bid']
        if fid in bid2buffer:
          boxs = np.concatenate([bid2buffer[fid][1], boxs], 0)
          buffers.append({'bid': bid, 'boxs': boxs, 'start': bid2buffer[fid][0]})
          del bid2buffer[fid]
        else:
          buffers.append({'bid': bid, 'boxs': boxs, 'start': f*8})
      for bid in bid2buffer:
        tracklets.append(bid2buffer[bid])
    for d in buffers:
      tracklets.append((d['start'], d['boxs']))
    # print name, num_frame, len(tracklets)
    # print [d[0] for d in tracklets]
    out_file = os.path.join(track_root_dir, name, 'merge.track')
    with open(out_file, 'w') as fout:
      for start, boxs in tracklets:
        fout.write('%d '%start)
        for box in boxs:
          for d in box:
            fout.write('%d '%int(d))
        fout.write('\n')


def viz_tracklet():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  track_root_dir = os.path.join(root_dir, 'kcf_track')
  gif_dir = os.path.join(root_dir, 'gif')
  viz_dir = os.path.join(root_dir, 'kcf_viz')

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
    imgs = []
    for i in range(len(gif)):
      img = np.asarray(gif[i][:, :, :3], dtype=np.uint8)
      imgs.append(img[:, :, ::-1].copy())
    h, w, _ = imgs[0].shape

    track_file = os.path.join(track_root_dir, name, 'merge.track')
    if not os.path.exists(track_file):
      continue
    with open(track_file) as f:
      cnt = 0
      for line in f:
        line = line.strip()
        data = line.split(' ')
        start = int(data[0])
        boxs = data[1:]
        # avg_area = 0.
        # for i in range(0, len(boxs), 4):
        #   avg_area += int(boxs[i+2]) * int(boxs[i+3])
        # avg_area /= len(boxs)/4
        # if avg_area >= h*w*2/3:
        #   continue
        for i in range(0, len(boxs), 4):
          frame = start + i/4
          x, y, w, h = [int(d) for d in boxs[i:i+4]]
          img = imgs[frame]
          cv2.rectangle(img, (x, y), (x+w, y+h), colormap12[cnt%len(colormap12)], 2);
        cnt += 1

    out_imgs = []
    for img in imgs:
      out_imgs.append(img[:, :, ::-1])
    out_file = os.path.join(viz_dir, name + '.merge.gif')
    imageio.mimsave(out_file, out_imgs)


def build_association_graph():
  # root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  # lst_file = os.path.join(root_dir, 'split.0.lst')
  # root_dir = '/home/jiac/data/tgif' # gpu8
  # lst_file = os.path.join(root_dir, 'split.3.lst')
  # track_root_dir = os.path.join(root_dir, 'kcf_track')

  root_dir = '/mnt/data2/jiac/vtt_raw' # neptune
  # lst_file = os.path.join(root_dir, '16.lst')
  # track_root_dir = os.path.join(root_dir, '16_kcf_track')
  # lst_file = os.path.join(root_dir, '17.lst')
  # track_root_dir = os.path.join(root_dir, '17_kcf_track')
  lst_file = os.path.join(root_dir, '18.lst')
  track_root_dir = os.path.join(root_dir, '18_kcf_track')

  gap = 8
  iou_threshold = 0.5

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

  # debug_set = set(['tumblr_nqlr0rn8ox1r2r0koo1_400'])
  # for name, num_frame in name_frames[:100]:
  for name, num_frame in name_frames:
    track_dir = os.path.join(track_root_dir, name)
    edges = []
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
        scores = np.zeros((num_forward, num_backward))
        edges.append(scores)
        continue

      forward_scores = np.mean(forward_scores, 1)
      backward_scores = np.mean(backward_scores, 1)
      scores = np.expand_dims(forward_scores, 1) + np.expand_dims(backward_scores, 0)
      scores = scores / 2.

      intersect_volumes = np.zeros((num_forward, num_backward))
      union_volumes = np.zeros((num_forward, num_backward))
      for i in range(gap):
        intersect = bbox_intersect(forward_boxs[:, i], backward_boxs[:, i]) # (num_forward, num_backward)
        intersect_volumes += intersect
        union = bbox_union(forward_boxs[:, i], backward_boxs[:, i])
        union_volumes += union
      ious = intersect_volumes / union_volumes
      valid = ious >= iou_threshold
      scores += ious
      scores = np.where(valid, scores, np.zeros(scores.shape))
      edges.append(scores)
    if len(edges) == 0:
      continue

    out_file = os.path.join(track_root_dir, name + '.viterbi')
    with open(out_file, 'w') as fout:
      while True:
        max_sum, path = viterbi_decoding(edges)
        if max_sum < iou_threshold:
          break
        for t, id in path:
          fout.write('%d,%d '%(t, id))
        fout.write('\n')
        remove_path_node_from_graph(edges, path)


def viz_viterbi_path():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'split.0.lst')
  gif_dir = os.path.join(root_dir, 'gif')
  track_root_dir = os.path.join(root_dir, 'kcf_track')
  viz_dir = os.path.join(root_dir, 'kcf_viz')

  gap = 8

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

  debug_set = set([
    'tumblr_nqq3ibcw841rvg72ao1_400', 
    'tumblr_npw7v7W07C1tmj047o1_250',
    'tumblr_n0lv33BBrb1rc0kvpo1_r1_250'
  ])

  alphas = np.arange(gap) / float(gap-1)
  alphas = np.expand_dims(alphas, 1)
  for name, num_frame in name_frames[:100]:
    # if name not in debug_set:
    #   continue
    print name, num_frame
    gif_file = os.path.join(gif_dir, name + '.gif')
    if not os.path.exists(gif_file):
      continue
    gif = imageio.mimread(gif_file, memtest=False)
    if len(gif[0].shape) < 3:
      continue
    imgs = []
    for i in range(len(gif)):
      img = np.asarray(gif[i][:, :, :3], dtype=np.uint8)
      imgs.append(img[:, :, ::-1].copy())

    path_file = os.path.join(track_root_dir, name + '.viterbi')
    # path_file = os.path.join(track_root_dir, name + '.viterbi.refine')
    paths = []
    with open(path_file) as f:
      for line in f:
        line = line.strip()
        data = line.split(' ')
        path = []
        for d in data:
          fields = d.split(',')
          path.append((int(fields[0]), int(fields[1])))
        paths.append(path)

    track_dir = os.path.join(track_root_dir, name)
    all_forward_boxs = []
    all_backward_boxs = []
    for f in range(0, num_frame, gap):
      forward_file = os.path.join(track_dir, '%d.track'%f)
      backward_file = os.path.join(track_dir, '%d.rtrack'%f)

      if os.path.exists(forward_file):
        forward_boxs, forward_scores = load_track(forward_file)
        all_forward_boxs.append(forward_boxs)
      if os.path.exists(backward_file):
        backward_boxs, backward_scores = load_track(backward_file, reverse=True)
        all_backward_boxs.append(backward_boxs)
    cnt = 0
    for path in paths:
      num_step = len(path)
      for i in range(num_step-1):
        step = path[i][0]
        fid = path[i][1]
        bid = path[i+1][1]
        if step < len(all_backward_boxs):
          boxes = all_forward_boxs[step][fid] * (1. - alphas) + all_backward_boxs[step][bid] * alphas
        else:
          boxes = all_forward_boxs[step][fid]
        for j in range(min(gap, boxes.shape[0])):
          f = step * gap + j
          if f >= len(imgs):
            continue 
          x, y, w, h = [int(d) for d in boxes[j]]
          cv2.rectangle(imgs[f], (x, y), (x+w, y+h), colormap12[cnt%len(colormap12)], 2);
      cnt += 1
    out_imgs = []
    for img in imgs:
      out_imgs.append(img[:, :, ::-1])
    out_file = os.path.join(viz_dir, name + '.viterbi.gif')
    # out_file = os.path.join(viz_dir, name + '.viterbi.refine.gif')
    imageio.mimsave(out_file, out_imgs)


def refine_viterbi_path():
  # root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  # lst_file = os.path.join(root_dir, 'split.0.lst')
  # root_dir = '/home/jiac/data/tgif' # gpu8
  # lst_file = os.path.join(root_dir, 'split.3.lst')
  # track_root_dir = os.path.join(root_dir, 'kcf_track')

  root_dir = '/mnt/data2/jiac/vtt_raw' # neptune
  # lst_file = os.path.join(root_dir, '16.lst')
  # track_root_dir = os.path.join(root_dir, '16_kcf_track')
  # lst_file = os.path.join(root_dir, '17.lst')
  # track_root_dir = os.path.join(root_dir, '17_kcf_track')
  lst_file = os.path.join(root_dir, '18.lst')
  track_root_dir = os.path.join(root_dir, '18_kcf_track')

  gap = 8
  max_num_step =  4
  maxmin_ratio_threshold = 2

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

  # debug_set = set([
  #   'tumblr_nqq3ibcw841rvg72ao1_400', 
  #   'tumblr_npw7v7W07C1tmj047o1_250',
  #   'tumblr_n0lv33BBrb1rc0kvpo1_r1_250'
  # ])

  alphas = np.arange(gap) / float(gap-1)
  alphas = np.expand_dims(alphas, 1)
  # for name, num_frame in name_frames[:100]:
  for name, num_frame in name_frames:
    # if name not in debug_set:
    #   continue

    path_file = os.path.join(track_root_dir, name + '.viterbi')
    if not os.path.exists(path_file):
      continue
    paths = []
    with open(path_file) as f:
      for line in f:
        line = line.strip()
        data = line.split(' ')
        path = []
        for d in data:
          fields = d.split(',')
          path.append((int(fields[0]), int(fields[1])))
        paths.append(path)

    track_dir = os.path.join(track_root_dir, name)
    all_forward_boxs = []
    all_backward_boxs = []
    for f in range(0, num_frame, gap):
      forward_file = os.path.join(track_dir, '%d.track'%f)
      backward_file = os.path.join(track_dir, '%d.rtrack'%f)
      if not os.path.exists(backward_file):
        continue

      forward_boxs, forward_scores = load_track(forward_file)
      backward_boxs, backward_scores = load_track(backward_file, reverse=True)
      all_forward_boxs.append(forward_boxs)
      all_backward_boxs.append(backward_boxs)
    refined_paths = []
    for path in paths:
      num_step = len(path)
      all_boxs = calc_boxs_for_path(path, all_forward_boxs, all_backward_boxs, alphas)
      cut_balance_maxmin_ratio(
        path, all_boxs, 0, num_step-1,max_num_step, maxmin_ratio_threshold, refined_paths)
    out_file = os.path.join(track_root_dir, name + '.viterbi.refine')
    with open(out_file, 'w') as fout:
      for path in refined_paths:
        for t, id in path:
          fout.write('%d,%d '%(t, id))
        fout.write('\n')


def viz_viterbi_path_vtt():
  root_dir = '/mnt/data2/jiac/vtt_raw' # neptune
  lst_file = os.path.join(root_dir, '16.lst')
  video_dir = os.path.join(root_dir, '16')
  track_root_dir = os.path.join(root_dir, '16_kcf_track')
  viz_dir = os.path.join(root_dir, '16_kcf_viz')

  gap = 8

  name_frames = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      name_frames.append((name, num_frame))

  alphas = np.arange(gap) / float(gap-1)
  alphas = np.expand_dims(alphas, 1)
  for name, num_frame in name_frames:
    video_file = os.path.join(video_dir, name + '.mp4')
    if not os.path.exists(video_file):
      continue
    vid = cv2.VideoCapture(video_file)
    fps = int(vid.get(cv2.cv.CV_CAP_PROP_FPS))
    imgs = []
    while True:
      flag, img = vid.read()
      if not flag:
        break
      imgs.append(img)

    path_file = os.path.join(track_root_dir, name + '.viterbi.refine')
    paths = []
    with open(path_file) as f:
      for line in f:
        line = line.strip()
        data = line.split(' ')
        path = []
        for d in data:
          fields = d.split(',')
          path.append((int(fields[0]), int(fields[1])))
        paths.append(path)

    track_dir = os.path.join(track_root_dir, name)
    all_forward_boxs = []
    all_backward_boxs = []
    for f in range(0, num_frame, gap):
      forward_file = os.path.join(track_dir, '%d.track'%f)
      backward_file = os.path.join(track_dir, '%d.rtrack'%f)

      if os.path.exists(forward_file):
        forward_boxs, forward_scores = load_track(forward_file)
        all_forward_boxs.append(forward_boxs)
      if os.path.exists(backward_file):
        backward_boxs, backward_scores = load_track(backward_file, reverse=True)
        all_backward_boxs.append(backward_boxs)
    cnt = 0
    for path in paths:
      num_step = len(path)
      for i in range(num_step-1):
        step = path[i][0]
        fid = path[i][1]
        bid = path[i+1][1]
        if step < len(all_backward_boxs):
          boxes = all_forward_boxs[step][fid] * (1. - alphas) + all_backward_boxs[step][bid] * alphas
        else:
          boxes = all_forward_boxs[step][fid]
        for j in range(min(gap, boxes.shape[0])):
          f = step * gap + j
          if f >= len(imgs):
            continue 
          x, y, w, h = [int(d) for d in boxes[j]]
          cv2.rectangle(imgs[f], (x, y), (x+w, y+h), colormap12[cnt%len(colormap12)], 2);
      cnt += 1
    out_file = os.path.join(viz_dir, name + '.mp4')
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    h, w, _ = imgs[0].shape
    vid = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
    for img in imgs:
      vid.write(img)


if __name__ == '__main__':
  # prepare_num_frame_lst()
  # prepare_num_frame_lst_vtt()
  # viz_tracking()
  # kcf_tracking()
  # viz_kcf_tracking()

  # associate_forward_backward()
  # viz_association()

  # generate_tracklet()
  # viz_tracklet()

  # build_association_graph()
  # refine_viterbi_path()
  # viz_viterbi_path()
  viz_viterbi_path_vtt()

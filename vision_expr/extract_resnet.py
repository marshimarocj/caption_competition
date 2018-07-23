import os
import json

import cv2
import imageio
import numpy as np
import mxnet as mx


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


def change_device(arg_params, aux_params, ctx):
  new_args = dict()
  new_auxs = dict()
  for k, v in arg_params.items():
    new_args[k] = v.as_in_context(ctx)
  for k, v in aux_params.items():
    new_auxs[k] = v.as_in_context(ctx)
  return new_args, new_auxs


'''expr
'''
def extract_vtt():
  root_dir = '/mnt/data2/jiac/vtt_raw' # neptune
  lst_file = os.path.join(root_dir, '18.lst')
  video_dir = os.path.join(root_dir, '18')
  track_dir = os.path.join(root_dir, '18_kcf_track')
  model_prefix = '/home/jiac/models/mxnet/resnet/11k/resnet-200/fullconv-resnet-imagenet-200-0'
  model_epoch = 22
  out_dir = os.path.join(root_dir, '18_track_ft', 'resnet200')

  ctx = mx.gpu(0)
  symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
  arg_params, aux_params = change_device(arg_params, aux_params, ctx)
  outs = symbol.get_internals()
  net = mx.symbol.Group([outs['pool1_output']])
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
      crop_imgs = []
      for d in path[::16]:
        frame = d['frame']
        if frame >= len(imgs):
          break

        cx = d['x']+ d['w']/2
        cy = d['y'] + d['h']/2
        crop_size = max(d['w'], d['h'])
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

        crop_img = cv2.resize(crop_img, (240, 240))
        crop_img = crop_img[:, :, ::-1] # rgb
        crop_img = np.moveaxis(crop_img, [0, 1, 2], [1, 2, 0])
        crop_imgs.append(crop_img)
      crop_imgs = np.array(crop_imgs, dtype=np.float32)
      # print crop_imgs.shape
      arg_params['data'] = mx.nd.array(crop_imgs, ctx)
      arg_params['pool1_output'] = mx.nd.empty((1,), ctx)
      exe = net.bind(ctx, arg_params, args_grad=None, grad_req='null', aux_states=aux_params)
      exe.forward(is_train=False)

      ft = exe.outputs[0].asnumpy()
      for d in [3, 2, 0]:
        ft = np.mean(ft, d)
      out_fts.append(ft)
    out_file = os.path.join(out_dir, name + '.npy')
    np.save(out_file, out_fts)


def extract_tgif():
  # root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  # lst_file = os.path.join(root_dir, 'split.0.lst')
  root_dir = '/home/jiac/data/tgif' # gpu8
  lst_file = os.path.join(root_dir, 'split.3.lst')
  gif_dir = os.path.join(root_dir, 'gif')
  track_dir = os.path.join(root_dir, 'kcf_track')
  model_prefix = '/home/jiac/models/mxnet/resnet/11k/resnet-200/fullconv-resnet-imagenet-200-0'
  model_epoch = 22
  out_dir = os.path.join(root_dir, 'track_ft', 'resnet200')

  ctx = mx.gpu(0)
  symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
  arg_params, aux_params = change_device(arg_params, aux_params, ctx)
  outs = symbol.get_internals()
  net = mx.symbol.Group([outs['pool1_output']])
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
      imgs.append(img.copy())
    img_h, img_w, _ = imgs[0].shape

    with open(track_file) as f:
      paths = json.load(f)
    out_fts = []
    for path in paths:
      crop_imgs = []
      for d in path[::16]:
        frame = d['frame']
        if frame >= len(imgs):
          break

        cx = d['x']+ d['w']/2
        cy = d['y'] + d['h']/2
        crop_size = max(d['w'], d['h'])
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

        crop_img = cv2.resize(crop_img, (240, 240))
        crop_img = np.moveaxis(crop_img, [0, 1, 2], [1, 2, 0])
        crop_imgs.append(crop_img)
      crop_imgs = np.array(crop_imgs, dtype=np.float32)
      # print crop_imgs.shape
      arg_params['data'] = mx.nd.array(crop_imgs, ctx)
      arg_params['pool1_output'] = mx.nd.empty((1,), ctx)
      exe = net.bind(ctx, arg_params, args_grad=None, grad_req='null', aux_states=aux_params)
      exe.forward(is_train=False)

      ft = exe.outputs[0].asnumpy()
      for d in [3, 2, 0]:
        ft = np.mean(ft, d)
      out_fts.append(ft)
    out_file = os.path.join(out_dir, name + '.npy')
    np.save(out_file, out_fts)


def extract_missing_tgif():
  root_dir = '/home/jiac/data2/tgif' # gpu9
  data_root_dir = os.path.join(root_dir, 'TGIF-Release/data')
  lst_file = os.path.join(root_dir, 'split.0.lst')
  valid_video_lst_file = os.path.join(root_dir, 'aux', 'int2video.npy')
  video_dir = os.path.join(data_root_dir, 'mp4')
  track_dir = os.path.join(data_root_dir, 'kcf_track')
  model_prefix = '/home/jiac/models/mxnet/resnet/11k/resnet-200/fullconv-resnet-imagenet-200-0'
  model_epoch = 22
  out_dir = os.path.join(data_root_dir, 'track_ft', 'resnet200')

  valid_videos = np.load(valid_video_lst_file)
  valid_videos = set(valid_videos.tolist())

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      name = data[0]
      num_frame = int(data[1])
      if num_frame > 0 and name in valid_videos:
        names.append(name)

  ctx = mx.gpu(0)
  symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
  arg_params, aux_params = change_device(arg_params, aux_params, ctx)
  outs = symbol.get_internals()
  net = mx.symbol.Group([outs['pool1_output']])
  print 'load complete'

  for name in names:
    video_file = os.path.join(video_dir, name + '.mp4')
    track_file = os.path.join(track_dir, name + '.json')
    out_file = os.path.join(out_dir, name + '.npy')
    if os.path.exists(out_file):
      continue

    imgs = load_video(video_file)
    print name, len(imgs)
    with open(track_file) as f:
      paths = json.load(f)
    out_fts = []
    for path in paths:
      crop_imgs = []
      for d in path[::16]:
        frame = d['frame']
        if frame >= len(imgs):
          break

        cx = d['x']+ d['w']/2
        cy = d['y'] + d['h']/2
        crop_size = max(d['w'], d['h'])
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

        crop_img = cv2.resize(crop_img, (240, 240))
        crop_img = crop_imgs[:, :, ::-1]
        crop_img = np.moveaxis(crop_img, [0, 1, 2], [1, 2, 0])
        crop_imgs.append(crop_img)
      crop_imgs = np.array(crop_imgs, dtype=np.float32)
      arg_params['data'] = mx.nd.array(crop_imgs, ctx)
      arg_params['pool1_output'] = mx.nd.empty((1,), ctx)
      exe = net.bind(ctx, arg_params, args_grad=None, grad_req='null', aux_states=aux_params)
      exe.forward(is_train=False)

      ft = exe.outputs[0].asnumpy()
      for d in [3, 2, 0]:
        ft = np.mean(ft, d)
      out_fts.append(ft)
    out_file = os.path.join(out_dir, name + '.npy')
    np.save(out_file, out_fts)


if __name__ == '__main__':
  # extract_vtt()
  # extract_tgif()
  extract_missing_tgif()

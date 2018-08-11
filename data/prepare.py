import os
import cPickle
import json
import re

import numpy as np

import bigfile


'''func
'''
def caption2id_mask(caption, max_words_in_caption, word2int):
  words = caption.split(' ')
  captionid = np.ones((max_words_in_caption,), dtype=np.int32)
  mask = np.zeros((max_words_in_caption,), dtype=np.int32)
  captionid[0] = 0
  mask[0] = 1
  for i, word in enumerate(words):
    if word in word2int:
      wid = word2int[word]
    else:
      wid = 2
    captionid[i+1] = wid
    mask[i+1] = 1

    if i+1 == max_words_in_caption-1:
      break
  i += 1
  if i+1 < max_words_in_caption:
    captionid[i+1] = 1
    mask[i+1] = 1
  return captionid, mask


def process_sent(sentence, allow_digits=True):
  sentence = sentence.replace(';', ',')
  sentence = sentence.lower()
  segments = sentence.split(',')
  output = []
  for segment in segments:
    if allow_digits:
      words = re.findall(r"['\-\w]+", segment)
    else:
      words = re.findall(r"['\-A-Za-z]+", segment)
    output.append(' '.join(words))
  output = ' , '.join(output)
  return output


def prepare_sa_feature(ft_file, dim_ft, num_step):
  if not os.path.exists(ft_file):
    ft = np.zeros((num_step, dim_ft), dtype=np.float32)
    mask = np.zeros((num_step,), dtype=np.float32)
  else:
    ft = np.load(ft_file)
    mask = np.zeros((num_step,), dtype=np.float32)
    num_ft, dim_ft = ft.shape
    mask[:min(num_ft, num_step)] = 1.
    if num_ft > num_step:
      ft = ft[:num_step]
    elif num_ft < num_step:
      ft = np.concatenate([ft, np.zeros((num_step-num_ft, dim_ft), dtype=np.float32)], 0)
  return ft, mask


def prepare_mp_feature(ft_file, dim_ft):
  if not os.path.exists(ft_file):
    ft = np.zeros((dim_ft), dtype=np.float32)
  else:
    ft = np.load(ft_file)
    ft = np.mean(ft, 0)
  return ft


'''expr
'''
def merge_tgif_trecvid16_rank_trn():
  # trecvid_root_dir = '/data1/jiac/trecvid2017/rank' # mercurial
  # tgif_root_dir = '/data1/jiac/tgif'
  # out_root_dir = '/data1/jiac/trecvid2018/rank'
  trecvid_root_dir = '/mnt/data1/jiac/trecvid2016' # neptune
  tgif_root_dir = '/mnt/data1/jiac/tgif'
  out_root_dir = '/mnt/data1/jiac/trecvid2018/rank'

  ##########ft#########
  tgif_ft_files = [
    'trn_ft.npy',
    'val_ft.npy',
    'tst_ft.npy',
    # 'trn_ft.max.npy',
    # 'val_ft.max.npy',
    # 'tst_ft.max.npy',
  ]
  trecvid_ft_files = [
    'tst_ft.npy',
    # 'tst_ft.max.npy',
  ]
  # for ft_name in ['i3d', 'resnet200']:
  for ft_name in ['i3d_flow']:
    print ft_name
    fts = []
    for tgif_ft_file in tgif_ft_files:
      file = os.path.join(tgif_root_dir, 'mp_feature', ft_name, tgif_ft_file)
      ft = np.load(file)
      fts.append(ft)
    for trecivd_ft_file in trecvid_ft_files:
      file = os.path.join(trecvid_root_dir, 'mp_feature', ft_name, trecivd_ft_file)
      ft = np.load(file)
      fts.append(ft)
    fts = np.concatenate(fts, 0)
    out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'trn_ft.npy')
    # out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'trn_ft.max.npy')
    np.save(out_file, fts)

  #########caption mask ##########
  # tgif_caption_mask_files = [
  #   os.path.join(tgif_root_dir, 'split', 'trn_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'val_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'tst_id_caption_mask.pkl'),
  # ]
  # trecvid_caption_mask_files = [
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.A.pkl'),
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.B.pkl'),
  # ]
  # idxs = []
  # caption_ids = []
  # caption_masks = []
  # base = 0
  # for file in tgif_caption_mask_files:
  #   with open(file) as f:
  #     data = cPickle.load(f)
  #   idxs.append(data[0] + base)
  #   caption_ids.append(data[1])
  #   caption_masks.append(data[2])
  #   base = np.max(data[0] + base) + 1
  # for file in trecvid_caption_mask_files:
  #   with open(file) as f:
  #     data = cPickle.load(f)
  #   idxs.append(data[0] + base)
  #   caption_ids.append(data[1])
  #   caption_masks.append(data[2])
  # idxs = np.concatenate(idxs, 0)
  # caption_ids = np.concatenate(caption_ids, 0)
  # caption_masks = np.concatenate(caption_masks, 0)
  # out_file = os.path.join(out_root_dir, 'split', 'trn_id_caption_mask.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump([idxs, caption_ids, caption_masks], fout)

  ##########vid##########
  # tgif_vid_files = [
  #   os.path.join(tgif_root_dir, 'split', 'trn_videoids.npy'),
  #   os.path.join(tgif_root_dir, 'split', 'val_videoids.npy'),
  #   os.path.join(tgif_root_dir, 'split', 'tst_videoids.npy'),
  # ]
  # vids = []
  # for file in tgif_vid_files:
  #   vids.append(np.load(file))
  # vids = np.concatenate(vids, 0)
  # base = np.max(vids) + 1
  # vids = [vids, range(base, base + 1915)]
  # vids = np.concatenate(vids)
  # out_file = os.path.join(out_root_dir, 'split', 'trn_videoids.npy')
  # np.save(out_file, vids)

  ##########caption_dict##########
  # tgif_caption_file = os.path.join(tgif_root_dir, 'annotation', 'human_caption_dict.pkl')
  # trecvid_caption_file = os.path.join(trecvid_root_dir, 'annotation', 'human_caption_dict.pkl')
  # vid2captions = {}
  # with open(tgif_caption_file) as f:
  #   data = cPickle.load(f)
  # vid2captions.update(data)
  # base = len(vid2captions)
  # with open(trecvid_caption_file) as f:
  #   data = cPickle.load(f)
  # start = len(data) - 1915
  # for vid in range(start, start + 1915):
  #   vid2captions[vid - start + base] = data[vid]
  # out_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump(vid2captions, fout)


def merge_tgif_trecvid16_rank_temporal_trn():
  trecvid_root_dir = '/mnt/data1/csz/trecvid16' # venus
  tgif_root_dir = '/mnt/data1/csz/TGIF'
  out_root_dir = '/mnt/data1/jiac/trecvid2018/rank'

  num_step = 20

  tgif_lst_files = [
    os.path.join(tgif_root_dir, 'public_split', 'trn_names.npy'),
    os.path.join(tgif_root_dir, 'public_split', 'val_names.npy'),
    os.path.join(tgif_root_dir, 'public_split', 'tst_names.npy'),
  ]
  ft_names = ['i3d.rgb', 'i3d.flow', 'resnet200']
  dim_fts = [1024, 1024, 2048]
  for ft_name, dim_ft in zip(ft_names, dim_fts):
    print ft_name
    fts = []
    masks = []

    splits = ['trn', 'val', 'tst']
    for s in range(3):
      lst_file = tgif_lst_files[s]
      split = splits[s]
      names = np.load(lst_file)
      for name in names:
        ft_file = os.path.join(tgif_root_dir, 'ordered_feature', 'raw', ft_name, split, '%s.npy'%name)
        ft, mask = prepare_sa_feature(ft_file, dim_ft, num_step)
        fts.append(ft)
        masks.append(mask)

    for vid in range(1, 1916):
      ft_file = os.path.join(trecvid_root_dir, 'ordered_feature', 'raw', ft_name, '%d.mp4.npy'%vid)
      ft, mask = prepare_sa_feature(ft_file, dim_ft, num_step)
      fts.append(ft)
      masks.append(mask)

    out_file = os.path.join(out_root_dir, 'temporal_ft', ft_name, 'trn.npz')
    np.savez_compressed(out_file, fts=fts, masks=masks)


def mscoco_rank_pretrain():
  mscoco_dir = '/data1/jiac/mscoco' # mercurial
  out_root_dir = '/data1/jiac/trecvid2018/rank'
  mscoco_word_file = os.path.join(mscoco_dir, 'aux', 'int2word.pkl')
  word_file = os.path.join(out_root_dir, 'annotation', 'int2word.pkl')

  ##########ft#########
  # mscoco_ft_files = [
  #   'trn_ft.npy',
  #   'val_ft.npy',
  #   'tst_ft.npy',
  # ]
  # ft_name = 'resnet200'
  # fts = []
  # for mscoco_ft_file in mscoco_ft_files:
  #   file = os.path.join(mscoco_dir, 'mp_feature', ft_name, mscoco_ft_file)
  #   ft = np.load(file)
  #   fts.append(ft)
  # fts = np.concatenate(fts, 0)
  # out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'pretrn_ft.npy')
  # np.save(out_file, fts)

  #########caption mask ##########
  mscoco_caption_mask_files = [
    os.path.join(mscoco_dir, 'split', 'trn_id_caption_mask.pkl'),
    os.path.join(mscoco_dir, 'split', 'val_id_caption_mask.pkl'),
    os.path.join(mscoco_dir, 'split', 'tst_id_caption_mask.pkl'),
  ]

  with open(mscoco_word_file) as f:
    mscoco_words = cPickle.load(f)

  with open(word_file) as f:
    words = cPickle.load(f)
  word2wid = {}
  for i, word in enumerate(words):
    word2wid[word] = i

  idxs = []
  caption_ids = []
  caption_masks = []
  base = 0
  for file in mscoco_caption_mask_files:
    with open(file) as f:
      data = cPickle.load(f)
    idxs.append(data[0] + base)
    captionid = np.concatenate([data[1], np.ones((data[1].shape[0], 10), dtype=np.int32)], 1)
    # caption_ids.append(captionid)
    for wids in captionid:
      for i in range(30):
        wid = wids[i]
        word = mscoco_words[wid]
        if word in word2wid:
          wid = word2wid[word]
        else:
          wid = 2
        wids[i] = wid
      caption_ids.append(wids)

    caption_mask = np.concatenate([data[2], np.zeros((data[2].shape[0], 10), dtype=np.bool_)], 1)
    caption_masks.append(caption_mask)
    base = np.max(data[0] + base) + 1
  idxs = np.concatenate(idxs, 0)
  caption_ids = np.array(caption_ids, dtype=np.int32)
  caption_masks = np.concatenate(caption_masks, 0)
  out_file = os.path.join(out_root_dir, 'split', 'pretrn_id_caption_mask.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump([idxs, caption_ids, caption_masks], fout)


def prepare_trecvid17_rank_val():
  root_dir = '/data1/jiac/trecvid2017' # mercurial
  out_root_dir = '/data1/jiac/trecvid2018/rank'
  word_file = os.path.join(out_root_dir, 'annotation', 'int2word.pkl')
  caption_files = [
    os.path.join(root_dir, 'VTT', 'matching.ranking.subtask', 'testing.2.subsets', 'tv17.vtt.descriptions.A'),
    os.path.join(root_dir, 'VTT', 'matching.ranking.subtask', 'testing.2.subsets', 'tv17.vtt.descriptions.B'),
  ]

  #######caption_mask#########
  # max_words_in_caption = 30

  # word2int = {}
  # with open(word_file) as f:
  #   words = cPickle.load(f)
  # for i, word in enumerate(words):
  #   word2int[word] = i

  # out_names = [
  #   'val_id_caption_mask.A.pkl',
  #   'val_id_caption_mask.B.pkl',
  # ]
  # for caption_file, out_name in zip(caption_files, out_names):
  #   idxs = []
  #   captionids = []
  #   caption_masks = []
  #   with open(caption_file) as f:
  #     for i, line in enumerate(f):
  #       line = line.strip()
  #       caption = process_sent(line)
  #       captionid, caption_mask = caption2id_mask(caption, max_words_in_caption, word2int)
  #       idxs.append(i)
  #       captionids.append(captionid)
  #       caption_masks.append(caption_mask)
  #   idxs = np.array(idxs, dtype=np.int32)
  #   captionids = np.array(captionids, dtype=np.int32)
  #   caption_masks = np.array(caption_masks, dtype=np.int32)
  #   out_file = os.path.join(out_root_dir, 'split', out_name)
  #   with open(out_file, 'w') as fout:
  #     cPickle.dump([idxs, captionids, caption_masks], fout)

  ########label#########
  label_file = os.path.join(root_dir, 'label', 'ranking', 'vtt.matching.ranking.set.2.gt')
  vid_file = os.path.join(root_dir, 'VTT', 'matching.ranking.subtask', 'testing.2.subsets', 'tv17.vtt.url.list')
  out_file = os.path.join(out_root_dir, 'label', '17.set.2.gt')

  vid2idx = {}
  with open(vid_file) as f:
    for i, line in enumerate(f):
      pos = line.find(' ')
      vid = int(line[:pos])
      vid2idx[vid] = i
  with open(label_file) as f, open(out_file, 'w') as fout:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      idx = vid2idx[int(data[0])]
      fout.write('%d %d %d\n'%(idx, int(data[1])-1, int(data[2])-1))


def prepare_trecvid17_temporal_val():
  trecvid_root_dir = '/mnt/data1/csz/trecvid17' # venus
  out_root_dir = '/mnt/data1/jiac/trecvid2018/rank'
  # lst_file = '/mnt/data1/jiac/trecvid2017/VTT/matching.ranking.subtask/testing.2.subsets/tv17.vtt.url.list'
  lst_file = '/mnt/data1/jiac/trecvid2017/VTT/description.generation.subtask/testing.URLs.video.description.subtask'

  num_step = 20

  vids = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      vid = int(data[0])
      vids.append(vid)

  ft_names = ['i3d.rgb', 'i3d.flow', 'resnet200']
  dim_fts = [1024, 1024, 2048]
  for ft_name, dim_ft in zip(ft_names, dim_fts):
    print ft_name
    fts = []
    masks = []

    for vid in vids:
      ft_file = os.path.join(trecvid_root_dir, 'ordered_feature', 'raw', ft_name, '%d.mp4.npy'%vid)
      ft, mask = prepare_sa_feature(ft_file, dim_ft, num_step)
      fts.append(ft)
      masks.append(mask)

    # out_file = os.path.join(out_root_dir, 'temporal_ft', ft_name, 'val.2.npz')
    out_file = os.path.join(out_root_dir, 'temporal_ft', ft_name, 'val.npz')
    np.savez_compressed(out_file, fts=fts, masks=masks)


def prepare_trecvid17_rank_gen_val():
  trecvid_root_dir = '/data1/jiac/trecvid2017' # mercurial
  out_root_dir = '/data1/jiac/trecvid2018/rank'
  word_file = os.path.join(trecvid_root_dir, 'rank', 'annotation', 'int2word.pkl')
  gt_file = os.path.join(trecvid_root_dir, 'label', 'description', 'trecvid17.json')

  ############caption_mask###########
  # max_words_in_caption = 30

  # word2int = {}
  # with open(word_file) as f:
  #   words = cPickle.load(f)
  # for i, word in enumerate(words):
  #   word2int[word] = i

  # with open(gt_file) as f:
  #   data = json.load(f)
  # num = len(data)
  # idxs = []
  # captionids = []
  # caption_masks = []
  # for vid in range(1, num+1):
  #   captions = data[str(vid)]
  #   for caption in captions:
  #     captionid, caption_mask = caption2id_mask(caption, max_words_in_caption, word2int)

  #     idxs.append(vid-1)
  #     captionids.append(captionid)
  #     caption_masks.append(caption_mask)
  # idxs = np.array(idxs, dtype=np.int32)
  # captionids = np.array(captionids, dtype=np.int32)
  # caption_masks = np.array(caption_masks, dtype=np.int32)

  # out_file = os.path.join(out_root_dir, 'split', 'val_id_caption_mask.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump([idxs, captionids, caption_masks], fout)

  #############caption_dict###########
  # vid2captions = {}
  # caption_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  # with open(caption_file) as f:
  #   vid2captions = cPickle.load(f)
  # base = 0
  # for vid in vid2captions:
  #   if vid > base:
  #     base = vid
  # base += 1
  # print len(vid2captions)

  # with open(gt_file) as f:
  #   data = json.load(f)
  # for vid in data:
  #   captions = data[vid]
  #   vid = int(vid) - 1 + base
  #   vid2captions[vid] = captions
  # print len(vid2captions)
  # # print vid2captions[105207]
  # out_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump(vid2captions, fout)

  ##########vid##########
  caption_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  out_file = os.path.join(out_root_dir, 'split', 'val_videoids.npy')

  with open(caption_file) as f:
    vid2captions = cPickle.load(f)
  max_vid = max(vid2captions.keys())
  vids = range(max_vid-1880+1, max_vid+1)
  np.save(out_file, vids)


def merge_tgif_trecvid17_gen_trn():
  trecvid_root_dir = '/data1/jiac/trecvid2017' # mercurial
  word_file = os.path.join(trecvid_root_dir, 'rank', 'annotation', 'int2word.pkl')
  gt_file = os.path.join(trecvid_root_dir, 'label', 'description', 'trecvid17.json')
  tgif_root_dir = '/data1/jiac/tgif'
  out_root_dir = '/data1/jiac/trecvid2018/generation'

  #############caption mask#########
  # max_words_in_caption = 30

  # tgif_caption_mask_files = [
  #   os.path.join(tgif_root_dir, 'split', 'trn_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'val_id_caption_mask.pkl'),
  #   os.path.join(tgif_root_dir, 'split', 'tst_id_caption_mask.pkl'),
  # ]
  # idxs = []
  # caption_ids = []
  # caption_masks = []
  # base = 0
  # for file in tgif_caption_mask_files:
  #   with open(file) as f:
  #     data = cPickle.load(f)
  #   idxs.append(data[0] + base)
  #   caption_ids.append(data[1])
  #   caption_masks.append(data[2])
  #   base = np.max(data[0] + base) + 1

  # word2int = {}
  # with open(word_file) as f:
  #   words = cPickle.load(f)
  # for i, word in enumerate(words):
  #   word2int[word] = i

  # with open(gt_file) as f:
  #   data = json.load(f)
  # num = len(data)
  # trecvid_idxs = []
  # trecvid_captionids = []
  # trecvid_caption_masks = []
  # for vid in range(1, num+1):
  #   captions = data[str(vid)]
  #   for caption in captions:
  #     captionid, caption_mask = caption2id_mask(caption, max_words_in_caption, word2int)

  #     trecvid_idxs.append(vid-1 + base)
  #     trecvid_captionids.append(captionid)
  #     trecvid_caption_masks.append(caption_mask)
  # trecvid_idxs = np.array(trecvid_idxs, dtype=np.int32)
  # trecvid_captionids = np.array(trecvid_captionids, dtype=np.int32)
  # trecvid_caption_masks = np.array(trecvid_caption_masks, dtype=np.int32)

  # idxs.append(trecvid_idxs)
  # caption_ids.append(trecvid_captionids)
  # caption_masks.append(trecvid_caption_masks)

  # idxs = np.concatenate(idxs, 0)
  # caption_ids = np.concatenate(caption_ids, 0)
  # caption_masks = np.concatenate(caption_masks, 0)
  # out_file = os.path.join(out_root_dir, 'split', 'trn_id_caption_mask.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump([idxs, caption_ids, caption_masks], fout)

  ##########vid##########
  # tgif_vid_files = [
  #   os.path.join(tgif_root_dir, 'split', 'trn_videoids.npy'),
  #   os.path.join(tgif_root_dir, 'split', 'val_videoids.npy'),
  #   os.path.join(tgif_root_dir, 'split', 'tst_videoids.npy'),
  # ]
  # vids = []
  # for file in tgif_vid_files:
  #   vids.append(np.load(file))
  # vids = np.concatenate(vids, 0)
  # base = np.max(vids) + 1
  # vids = [vids, range(base, base + 1880)]
  # vids = np.concatenate(vids)
  # out_file = os.path.join(out_root_dir, 'split', 'trn_videoids.npy')
  # np.save(out_file, vids)

  ##########ft#########
  # tgif_ft_files = [
  #   'trn_ft.npy',
  #   'val_ft.npy',
  #   'tst_ft.npy',
  # ]
  # trecvid_ft_file = 'val_ft.npy'
  # for ft_name in ['i3d', 'resnet200']:
  #   print ft_name
  #   fts = []
  #   for tgif_ft_file in tgif_ft_files:
  #     file = os.path.join(tgif_root_dir, 'mp_feature', ft_name, tgif_ft_file)
  #     ft = np.load(file)
  #     fts.append(ft)
  #   file = os.path.join('/data1/jiac/trecvid2018/rank', 'mp_feature', ft_name, trecvid_ft_file)
  #   ft = np.load(file)
  #   fts.append(ft)
  #   fts = np.concatenate(fts, 0)
  #   out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'trn_ft.npy')
  #   np.save(out_file, fts)

  ##########caption_dict##########
  tgif_caption_file = os.path.join(tgif_root_dir, 'annotation', 'human_caption_dict.pkl')
  vid2captions = {}
  with open(tgif_caption_file) as f:
    vid2captions = cPickle.load(f)
  base = len(vid2captions)
  with open(gt_file) as f:
    data = json.load(f)
  for vid in range(len(data)):
    vid2captions[vid + base] = data[str(vid+1)]
  out_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump(vid2captions, fout)


def prepare_trecvid16_gen_val():
  trecvid_root_dir = '/data1/jiac/trecvid2017/rank' # mercurial
  out_root_dir = '/data1/jiac/trecvid2018/generation'
  word_file = os.path.join(trecvid_root_dir, 'generation', 'annotation', 'int2word.pkl')

  #########caption mask ##########
  # trecvid_caption_mask_files = [
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.A.pkl'),
  #   os.path.join(trecvid_root_dir, 'split', 'tst_id_caption_mask.B.pkl'),
  # ]
  # idxs = []
  # caption_ids = []
  # caption_masks = []
  # for file in trecvid_caption_mask_files:
  #   with open(file) as f:
  #     data = cPickle.load(f)
  #   idxs.append(data[0])
  #   caption_ids.append(data[1])
  #   caption_masks.append(data[2])
  # idxs = np.concatenate(idxs, 0)
  # caption_ids = np.concatenate(caption_ids, 0)
  # caption_masks = np.concatenate(caption_masks, 0)
  # out_file = os.path.join(out_root_dir, 'split', 'val_id_caption_mask.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump([idxs, caption_ids, caption_masks], fout)

  #############caption_dict###########
  # vid2captions = {}
  # caption_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  # with open(caption_file) as f:
  #   vid2captions = cPickle.load(f)
  # base = len(vid2captions)
  # print len(vid2captions)

  # trecvid_caption_file = os.path.join(trecvid_root_dir, 'annotation', 'human_caption_dict.pkl')
  # base = len(vid2captions)
  # with open(trecvid_caption_file) as f:
  #   data = cPickle.load(f)
  # start = len(data) - 1915
  # for vid in range(start, start + 1915):
  #   vid2captions[vid - start + base] = data[vid]
  # print len(vid2captions)
  # out_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  # with open(out_file, 'w') as fout:
  #   cPickle.dump(vid2captions, fout)

  ###########vid###########
  caption_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  out_file = os.path.join(out_root_dir, 'split', 'val_videoids.npy')

  with open(caption_file) as f:
    vid2captions = cPickle.load(f)
  total = len(vid2captions)
  vids = range(total - 1915, total)
  np.save(out_file, vids)


def prepare_sbu_word2vec():
  root_dir = '/data1/jiac/trecvid2018' # mercurial
  voc_file = os.path.join(root_dir, 'rank', 'annotation', 'int2word.pkl')
  sbu_wvec_file = '/data1/syq/image-retrieval/vsepp/sbu_vocab.json'
  out_file = os.path.join(root_dir, 'rank', 'annotation', 'E.sbu.word2vec.npy')

  with open(voc_file) as f:
    words = cPickle.load(f)
  num_word = len(words)

  wvecs = np.zeros((num_word, 512), dtype=np.float32)

  with open(sbu_wvec_file) as f:
    word2vec = json.load(f)
  hit = 0
  for i, word in enumerate(words):
    if word in word2vec:
      wvecs[i] = np.array(word2vec[word], dtype=np.float32)
      hit += 1
  print hit, num_word

  np.save(out_file, wvecs)


def prepare_flickr30k_word2vec():
  root_dir = '/data1/jiac/trecvid2018' # mercurial
  voc_file = os.path.join(root_dir, 'rank', 'annotation', 'int2word.pkl')
  flickr30m_dir = '/home/qjin/data/word2vec/flickr/vec500flickr30m'
  out_file = os.path.join(root_dir, 'rank', 'annotation', 'E.flick30m.word2vec.npy')

  with open(voc_file) as f:
    words = cPickle.load(f)
  num_word = len(words)

  wvecs = np.zeros((num_word, 500), dtype=np.float32)

  bf = bigfile.BigFile(flickr30m_dir)
  for i, word in enumerate(words):
    if word in bf.name2index:
      vec = bf.read_one(word)
      wvecs[i] = np.array(vec, dtype=np.float32)

  np.save(out_file, wvecs)


def prepare_tgif_flow_ft():
  root_dir = '/mnt/data1/jiac/tgif' # neptune
  vid_file = os.path.join(root_dir, 'aux', 'int2video.npy')
  split_files = [
    os.path.join(root_dir, 'split', 'trn_videoids.npy'),
    os.path.join(root_dir, 'split', 'val_videoids.npy'),
    os.path.join(root_dir, 'split', 'tst_videoids.npy'),
  ]
  raw_ft_dir = '/mnt/data3/TGIF/ordered_feature/raw/i3d.flow'
  out_files = [
    # os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'trn_ft.npy'),
    # os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'val_ft.npy'),
    # os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.npy'),
    os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'trn_ft.max.npy'),
    os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'val_ft.max.npy'),
    os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.max.npy'),
  ]

  dim_ft = 1024

  videos = np.load(vid_file)

  for s in range(3):
    split_file = split_files[s]
    out_file = out_files[s]
    vids = np.load(split_file)
    print s

    out_fts = []
    for vid in vids:
      video = videos[vid]
      ft_file = os.path.join(raw_ft_dir, video + '.gif.npy')
      if not os.path.exists(ft_file):
        ft = np.zeros((dim_ft,), dtype=np.float32)
      else:
        fts = np.load(ft_file)
        # ft = np.mean(fts, 0)
        ft = np.max(fts, 0)
      out_fts.append(ft)
    out_fts = np.array(out_fts, dtype=np.float32)
    np.save(out_file, out_fts)


def prepare_trecvid16_flow_ft():
  root_dir = '/mnt/data1/jiac/trecvid2016' # neptune
  raw_ft_dir = '/mnt/data3/trecvid/ordered_feature/i3d.flow/16'
  # out_file = os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.npy')
  out_file = os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.max.npy')

  dim_ft = 1024

  out_fts = []
  for vid in range(1, 1916):
    ft_file = os.path.join(raw_ft_dir, '%d.mp4.npy'%vid)
    if not os.path.exists(ft_file):
      ft = np.zeros((dim_ft,), dtype=np.float32)
    else:
      fts = np.load(ft_file)
      # ft = np.mean(fts, 0)
      ft = np.max(fts, 0)
    out_fts.append(ft)
  out_fts = np.array(out_fts, dtype=np.float32)
  np.save(out_file, out_fts)


def prepare_trecvid17_flow_ft():
  root_dir = '/mnt/data1/jiac/trecvid2017' # neptune
  raw_ft_dir = '/mnt/data3/trecvid/ordered_feature/i3d.flow/17'

  lst_file = os.path.join(root_dir, 'VTT', 'matching.ranking.subtask', 'testing.2.subsets', 'tv17.vtt.url.list')
  out_file = os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.2.npy')
  # out_file = os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.2.max.npy')

  # lst_file = os.path.join(root_dir, 'VTT', 'description.generation.subtask', 'testing.URLs.video.description.subtask')
  # # out_file = os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.npy')
  # out_file = os.path.join(root_dir, 'mp_feature', 'i3d_flow', 'tst_ft.max.npy')

  dim_ft = 1024

  out_fts = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      vid = line[:pos]
      ft_file = os.path.join(raw_ft_dir, '%s.mp4.npy'%vid)
      if not os.path.exists(ft_file):
        ft = np.zeros((dim_ft,), dtype=np.float32)
      else:
        fts = np.load(ft_file)
        ft = np.mean(fts, 0)
        # ft = np.max(fts, 0)
      out_fts.append(ft)
  out_fts = np.array(out_fts, dtype=np.float32)
  np.save(out_file, out_fts)


def prepare_trecvid18_rank_tst():
  root_dir = '/data1/jiac/trecvid2018' # mercurial
  out_root_dir = '/data1/jiac/trecvid2018/rank'
  word_file = os.path.join(out_root_dir, 'annotation', 'int2word.pkl')
  caption_files = [
    os.path.join(root_dir, 'VTT', 'ranking', 'tv18.vtt.descriptions.A'),
    os.path.join(root_dir, 'VTT', 'ranking', 'tv18.vtt.descriptions.B'),
    os.path.join(root_dir, 'VTT', 'ranking', 'tv18.vtt.descriptions.C'),
    os.path.join(root_dir, 'VTT', 'ranking', 'tv18.vtt.descriptions.D'),
    os.path.join(root_dir, 'VTT', 'ranking', 'tv18.vtt.descriptions.E'),
  ]

  max_words_in_caption = 30

  word2int = {}
  with open(word_file) as f:
    words = cPickle.load(f)
  for i, word in enumerate(words):
    word2int[word] = i

  out_names = [
    'tst_id_caption_mask.A.pkl',
    'tst_id_caption_mask.B.pkl',
    'tst_id_caption_mask.C.pkl',
    'tst_id_caption_mask.D.pkl',
    'tst_id_caption_mask.E.pkl',
  ]
  for caption_file, out_name in zip(caption_files, out_names):
    idxs = []
    captionids = []
    caption_masks = []
    with open(caption_file) as f:
      for i, line in enumerate(f):
        line = line.strip()
        caption = process_sent(line)
        captionid, caption_mask = caption2id_mask(caption, max_words_in_caption, word2int)
        idxs.append(i)
        captionids.append(captionid)
        caption_masks.append(caption_mask)
    idxs = np.array(idxs, dtype=np.int32)
    captionids = np.array(captionids, dtype=np.int32)
    caption_masks = np.array(caption_masks, dtype=np.int32)
    out_file = os.path.join(out_root_dir, 'split', out_name)
    with open(out_file, 'w') as fout:
      cPickle.dump([idxs, captionids, caption_masks], fout)


def prepare_trecvid18_ft_tst():
  trecvid_root_dir = '/mnt/data1/csz/trecvid18' # venus
  out_root_dir = '/mnt/data1/jiac/trecvid2018/rank'

  ######temporal feature#####
  ft_names = ['i3d.rgb', 'resnet200']
  dim_fts = [1024, 2048]
  # ft_names = ['i3d.flow']
  # dim_fts = [1024]
  num_step = 20
  for ft_name, dim_ft in zip(ft_names, dim_fts):
    print ft_name
    fts = []
    masks = []

    for vid in range(1, 1921):
      ft_file = os.path.join(trecvid_root_dir, 'ordered_feature', 'raw', ft_name, '%d.mp4.npy'%vid)
      ft, mask = prepare_sa_feature(ft_file, dim_ft, num_step)
      fts.append(ft)
      masks.append(mask)

    out_file = os.path.join(out_root_dir, 'temporal_ft', ft_name, 'tst.npz')
    np.savez_compressed(out_file, fts=fts, masks=masks)

  ######mp feature######
  # # ft_names = ['i3d.rgb', 'resnet200']
  # # dim_fts = [1024, 2048]
  # ft_names = ['i3d.flow']
  # dim_fts = [1024]
  # for ft_name, dim_ft in zip(ft_names, dim_fts):
  #   fts = []

  #   for vid in range(1, 1921):
  #     ft_file = os.path.join(trecvid_root_dir, 'ordered_feature', 'raw', ft_name, '%d.mp4.npy'%vid)
  #     ft = prepare_mp_feature(ft_file, dim_ft)
  #     fts.append(ft)

  #   out_file = os.path.join(out_root_dir, 'mp_ft', ft_name, 'tst.npy')
  #   np.save(out_file, fts)


if __name__ == '__main__':
  # merge_tgif_trecvid16_rank_trn()
  merge_tgif_trecvid16_rank_temporal_trn()
  # prepare_trecvid17_temporal_val()
  # prepare_trecvid17_rank_val()
  # prepare_trecvid17_rank_gen_val()
  # prepare_trecvid18_rank_tst()
  # mscoco_rank_pretrain()

  # merge_tgif_trecvid17_gen_trn()
  # prepare_trecvid16_gen_val()

  # prepare_sbu_word2vec()
  # prepare_flickr30k_word2vec()

  # prepare_tgif_flow_ft()
  # prepare_trecvid16_flow_ft()
  # prepare_trecvid17_flow_ft()

  # prepare_trecvid18_ft_tst()

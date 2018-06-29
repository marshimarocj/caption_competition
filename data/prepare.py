import os
import cPickle
import json
import re

import numpy as np


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


'''expr
'''
def merge_tgif_trecvid16_rank_trn():
  trecvid_root_dir = '/data1/jiac/trecvid2017/rank' # mercurial
  tgif_root_dir = '/data1/jiac/tgif'
  out_root_dir = '/data1/jiac/trecvid2018/rank'

  ##########ft#########
  # tgif_ft_files = [
  #   'trn_ft.npy',
  #   'val_ft.npy',
  #   'tst_ft.npy',
  # ]
  # trecvid_ft_files = [
  #   'tst_ft.npy',
  # ]
  # for ft_name in ['i3d', 'resnet200']:
  #   print ft_name
  #   fts = []
  #   for tgif_ft_file in tgif_ft_files:
  #     file = os.path.join(tgif_root_dir, 'mp_feature', ft_name, tgif_ft_file)
  #     ft = np.load(file)
  #     fts.append(ft)
  #   for trecivd_ft_file in trecvid_ft_files:
  #     file = os.path.join(trecvid_root_dir, 'mp_feature', ft_name, trecivd_ft_file)
  #     ft = np.load(file)
  #     fts.append(ft)
  #   fts = np.concatenate(fts, 0)
  #   out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'trn_ft.npy')
  #   np.save(out_file, fts)

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
  tgif_caption_file = os.path.join(tgif_root_dir, 'annotation', 'human_caption_dict.pkl')
  trecvid_caption_file = os.path.join(trecvid_root_dir, 'annotation', 'human_caption_dict.pkl')
  vid2captions = {}
  with open(tgif_caption_file) as f:
    data = cPickle.load(f)
  vid2captions.update(data)
  base = len(vid2captions)
  with open(trecvid_caption_file) as f:
    data = cPickle.load(f)
  start = len(data) - 1915
  for vid in range(start, start + 1915):
    vid2captions[vid - start + base] = data[vid]
  out_file = os.path.join(out_root_dir, 'annotation', 'human_caption_dict.pkl')
  with open(out_file, 'w') as fout:
    cPickle.dump(vid2captions, fout)


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
  tgif_ft_files = [
    'trn_ft.npy',
    'val_ft.npy',
    'tst_ft.npy',
  ]
  trecvid_ft_file = 'val_ft.npy'
  for ft_name in ['i3d', 'resnet200']:
    print ft_name
    fts = []
    for tgif_ft_file in tgif_ft_files:
      file = os.path.join(tgif_root_dir, 'mp_feature', ft_name, tgif_ft_file)
      ft = np.load(file)
      fts.append(ft)
    file = os.path.join('/data1/jiac/trecvid2018/rank/mp_feature', 'mp_feature', ft_name, trecvid_ft_file)
    ft = np.load(file)
    fts.append(ft)
    fts = np.concatenate(fts, 0)
    out_file = os.path.join(out_root_dir, 'mp_feature', ft_name, 'trn_ft.npy')
    np.save(out_file, fts)


if __name__ == '__main__':
  # merge_tgif_trecvid16_rank_trn()
  # prepare_trecvid17_rank_val()
  # prepare_trecvid17_rank_gen_val()

  merge_tgif_trecvid17_gen_trn()

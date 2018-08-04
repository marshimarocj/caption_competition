import os
import cPickle
import json

from nltk.corpus import stopwords


'''func
'''


'''expr
'''
def gen_concept_wid():
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  word_file = os.path.join(root_dir, 'annotation', 'int2word.pkl')
  caption_file = os.path.join(root_dir, 'split', 'trn_id_caption_mask.pkl')
  out_file = os.path.join(root_dir, 'annotation', 'concept_wid.json')

  with open(word_file) as f:
    words = cPickle.load(f)

  stopword_set = set(stopwords.words('english'))
  wid2cnt = {}
  for i, word in enumerate(words):
    if i < 3:
      continue
    if word not in stopword_set:
      wid2cnt[i] = 0

  threshold = 100

  with open(caption_file) as f:
    ft_idxs, captionids, caption_masks = cPickle.load(f)
    for captionid in captionids:
      for wid in captionid:
        if wid == 1:
          break
        if wid in wid2cnt:
          wid2cnt[wid] += 1

  valid_wids = []
  for wid in wid2cnt:
    cnt = wid2cnt[wid]
    if cnt >= threshold:
      valid_wids.append(wid)

  print len(valid_wids)

  with open(out_file, 'w') as fout:
    json.dump(valid_wids, fout)


if __name__ == '__main__':
  gen_concept_wid()

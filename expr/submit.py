import os
import json

import numpy as np


'''func
'''


'''expr
'''
def submit_rerank():
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  pred_files = [
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.A.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.B.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.C.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.D.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.E.npy'),
  ]
  rerank_files = [
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.A.rerank.20.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.B.rerank.20.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.C.rerank.20.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.D.rerank.20.npy'),
    os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'tst.E.rerank.20.npy'),
  ]
  out_files = [
    os.path.join(root_dir, 'submit', 'group.align.A'),
    os.path.join(root_dir, 'submit', 'group.align.B'),
    os.path.join(root_dir, 'submit', 'group.align.C'),
    os.path.join(root_dir, 'submit', 'group.align.D'),
    os.path.join(root_dir, 'submit', 'group.align.E'),
  ]

  for pred_file, rerank_file, out_file in zip(pred_files, rerank_files, out_files):
    predicts = np.load(pred_file)
    rerank_predicts = np.load(rerank_file)
    combined_predicts = .5 * predicts + .5 * rerank_predicts

    num_txt = combined_predicts.shape[1]

    with open(out_file, 'w') as fout:
      for i, predict in enumerate(combined_predicts):
        sort_idxs = np.argsort(-predict)
        for j, idx in enumerate(sort_idxs):
          fout.write('%d %d %d\n'%(j+1, i+1, idx+1))
      for j in range(num_txt):
        fout.write('%d 1921 %d\n'%(j+1, j+1))


def submit_generation():
  root_dir = '/home/jiac/data/trecvid18/vtt' # earth
  files = [
    os.path.join(root_dir, 'description', 'tst', 'generation', 'cj_ensemble.json'),
    os.path.join(root_dir, 'description', 'tst', 'generation', 'diversity.json'),
  ]
  out_files = [
    os.path.join(root_dir, 'submit', 'generation', 'vevd_ensemble'),
    os.path.join(root_dir, 'submit', 'generation', 'diversity'),
  ]

  for file, out_file in zip(files, out_files):
    with open(file) as f:
      data = json.load(f)
    with open(out_file, 'w') as fout:
      fout.write('runType=V\n')
      for d in data:
        fout.write('%d %s\n'%(d['vid'], d['caption']))


if __name__ == '__main__':
  # submit_rerank()
  submit_generation()

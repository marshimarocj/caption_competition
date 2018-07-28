import os
import json


'''func
'''


'''expr
'''
def viz_generation():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune

  # predict_file = os.path.join(root_dir, 'generation', 'vead_expr', 'i3d_resnet200.512.512', 'pred', 'val-180.1.5.beam.json')
  # out_file = os.path.join(root_dir, 'generation', 'vead_expr', 'i3d_resnet200.512.512', 'pred', 'viz.json')

  # predict_file = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'epoch-136.1.5.beam.json')
  # out_file = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm', 'pred', 'viz.json')

  # predict_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'val-89.1.5.beam.json')
  # out_file = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr', 'pred', 'viz.json')

  # predict_file = os.path.join(root_dir, 'generation', 'margin_expr', 'i3d_resnet200.512.512.0.5.16.5.0.1.cider', 'pred', 'val-75.1.5.beam.json')
  # out_file = os.path.join(root_dir, 'generation', 'margin_expr', 'i3d_resnet200.512.512.0.5.16.5.0.1.cider', 'pred', 'viz.json')

  predict_file = os.path.join(root_dir, 'generation', 'self_critique_expr', 'i3d_resnet200.512.512.bcmr', 'pred', 'val-88.1.5.beam.json')
  out_file = os.path.join(root_dir, 'generation', 'self_critique_expr', 'i3d_resnet200.512.512.bcmr', 'pred', 'viz.json')

  with open(predict_file) as f:
    data = json.load(f)
  min_key = min([int(d) for d in data.keys()])

  out = []
  for key in data:
    caption = data[key]
    id = int(key) - min_key + 1
    out.append({'vid': id, 'caption': caption})
  out = sorted(out, key=lambda x:x['vid'])

  with open(out_file, 'w') as fout:
    json.dump(out, fout, indent=2)


def viz_rank():
  root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  caption_files = [
    '/data1/jiac/trecvid2017/VTT/matching.ranking.subtask/testing.2.subsets/tv17.vtt.descriptions.A',
    '/data1/jiac/trecvid2017/VTT/matching.ranking.subtask/testing.2.subsets/tv17.vtt.descriptions.B',
  ]
  vid_file = '/data1/jiac/trecvid2017/VTT/matching.ranking.subtask/testing.2.subsets/tv17.vtt.url.list'
  pred_files = [
    [
      os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.A.npy'),
      os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    ],
    [
      os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.B.npy'),
      os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.B.rerank.20.npy'),
    ],
  ]
  out_files = [
    os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'viz.A.json'),
    os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'viz.B.json'),
  ]

  vids = []
  with open(vid_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      vid = int(line[:pos])
      vids.append(vid)

  caption_sets = []
  for caption_file in caption_files:
    captions = []
    with open(caption_file) as f:
      for line in f:
        line = line.strip()
        pos = line.find(' ')
        captions.append(line[pos+1:])

  for pred_file, out_file, captions in zip(pred_files, out_files, caption_sets):
    preds = np.load(pred_file[0])
    preds += np.load(pred_file[1])
    sort_idxs = np.argsort(-preds, 1)
    out = []
    for vid, idxs, pred in zip(vids, sort_idxs, preds):
      out.append({
        'vid': vid,
        'captions': [
          {
            'caption': captions[idxs[0]],
            'score': float(pred[idx[0]]) / 2.
          },
          {
            'caption': captions[idxs[1]],
            'score': float(pred[idx[1]]) / 2.
          },
          {
            'caption': captions[idxs[2]],
            'score': float(pred[idx[2]]) / 2.
          },
        ]
      })
    print out_file
    with open(out_file, 'w') as fout:
      json.dump(out, fout)


if __name__ == '__main__':
  # viz_generation()
  viz_rank()

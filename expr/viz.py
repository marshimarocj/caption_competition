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

  predict_file = os.path.join(root_dir, 'generation', 'margin_expr', 'i3d_resnet200.512.512.0.5.16.5.0.1.cider', 'pred', 'val-75.1.5.beam.json')
  out_file = os.path.join(root_dir, 'generation', 'margin_expr', 'i3d_resnet200.512.512.0.5.16.5.0.1.cider', 'pred', 'viz.json')

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


if __name__ == '__main__':
  viz_generation()

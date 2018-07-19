import os

import framework.util.graph_ckpt


'''func
'''


'''expr
'''
def export_rank_video_embed():
  root_dir = '/data1/jiac/trecvid2018' # uranus
  model_file = os.path.join(root_dir, 'rank', 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'model', 'epoch-77')

  name2var = framework.util.graph_ckpt.load_variable_in_ckpt(model_file)
  print name2var.keys()


if __name__ == '__main__':
  export_rank_video_embed()

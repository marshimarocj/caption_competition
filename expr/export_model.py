import os
import sys
sys.path.append('../')

import numpy as np

import framework.util.graph_ckpt
import model.rnnve_orth


'''func
'''


'''expr
'''
def export_rank_video_embed():
  root_dir = '/data1/jiac/trecvid2018' # uranus
  # root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  # model_file = os.path.join(root_dir, 'rank', 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'model', 'epoch-77')
  # out_file = os.path.join(root_dir, 'rank', 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'model', 'ft_embed_weight.npz')
  # model_file = os.path.join(root_dir, 'generation', 'vead_expr', 'i3d_resnet200.512.512', 'model', 'epoch-173')
  model_file = os.path.join(root_dir, 'rank', 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'model', 'epoch-41')

  name2var = framework.util.graph_ckpt.load_variable_in_ckpt(model_file)
  print name2var.keys()
  # ft_pca_W = name2var['rnnve.Model/ft_pca_W']
  # ft_pca_B = name2var['rnnve.Model/ft_pca_B']
  # np.savez_compressed(out_file, ft_pca_W=ft_pca_W, ft_pca_B=ft_pca_B)


if __name__ == '__main__':
  export_rank_video_embed()

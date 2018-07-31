import os
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf

import framework.util.graph_ckpt
import rank_model.rnnve_orth
import rank_driver.rnnve_orth


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


def init_orth_model():
  # root_dir = '/data1/jiac/trecvid2018' # uranus
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  model_file = os.path.join(root_dir, 'rank', 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'model', 'epoch-41')
  expr_name = os.path.join(root_dir, 'rank', 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m')
  model_cfg_file = '%s.model.json'%expr_name
  path_cfg_file = '%s.path.json'%expr_name
  out_file = os.path.join(expr_name, 'model', 'pretrain')

  path_cfg = rank_driver.rnnve_orth.gen_dir_struct_info(path_cfg_file)
  path_cfg.model_file = ''
  model_cfg = rank_driver.rnnve_orth.load_and_fill_model_cfg(model_cfg_file, path_cfg)

  key_map = {
    'rnn.GRUCell/candidate_b': 'rnn.GRUCell/candidate_b',
    'rnn.GRUCell/candidate_W': 'rnn.GRUCell/candidate_W',
    'rnn.GRUCell/gate_b': 'rnn.GRUCell/gate_b',
    'rnn.GRUCell/gate_W': 'rnn.GRUCell/gate_W',
    'rnn.GRUCell.reverse/candidate_b': 'rnn.GRUCell.reverse/candidate_b',
    'rnn.GRUCell.reverse/candidate_W': 'rnn.GRUCell.reverse/candidate_W',
    'rnn.GRUCell.reverse/gate_b': 'rnn.GRUCell.reverse/gate_b',
    'rnn.GRUCell.reverse/gate_W': 'rnn.GRUCell.reverse/gate_W',
    'word.Encoder/word_embedding_W': 'word.Encoder/word_embedding_W',
  }

  m = rank_model.rnnve_orth.Model(model_cfg)
  trn_tst_graph = m.build_trn_tst_graph(decay_boundarys=[])
  with trn_tst_graph.as_default():
    var_names = [v.op.name for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)]
    print var_names
    assign_op, feed_dict = framework.util.graph_ckpt.init_weight_from_singlemodel(model_file, key_map)

  with tf.Session(graph=trn_tst_graph) as sess:
    sess.run(m.init_op)
    sess.run(assign_op, feed_dict=feed_dict)
    m.saver.save(sess, out_file)


if __name__ == '__main__':
  # export_rank_video_embed()
  init_orth_model()

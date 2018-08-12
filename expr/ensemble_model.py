import os
import json
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf

import framework.util.graph_ckpt
import gen_model.vevd
import gen_driver.vevd


'''func
'''
def select_best_epoch(log_dir):
  names = os.listdir(log_dir)
  best_cider = 0.
  best_epoch = -1
  for name in names:
    if 'val_metrics' in name:
      file = os.path.join(log_dir, name)
      with open(file) as f:
        data = json.load(f)
      cider = data['cider']
      epoch = data['epoch']
      if cider > best_cider:
        best_cider = cider
        best_epoch = epoch
  return best_epoch, best_cider


'''expr
'''
def export_avg_model_weights():
  root_dir = '/mnt/data1/jiac/trecvid2018' # neptune
  # expr_name = os.path.join(root_dir, 'generation', 'vevd_expr', 'i3d_resnet200_i3d_flow.512.512.lstm')
  # expr_name = os.path.join(root_dir, 'rank', 'vevd_expr', 'i3d_resnet200.512.512.lstm')
  # expr_name = os.path.join(root_dir, 'generation', 'self_critique_expr', 'i3d_resnet200.512.512.bcmr')
  # expr_name = os.path.join(root_dir, 'generation', 'diversity_expr', 'i3d_resnet200.512.512.0.2.5.2_4.bcmr')
  expr_name = os.path.join(root_dir, 'generation', 'margin_expr', 'i3d_resnet200.512.512.0.5.16.5.0.1.cider')
  log_dir = os.path.join(expr_name, 'log')
  model_cfg_file = '%s.model.json'%expr_name
  path_cfg_file = '%s.path.json'%expr_name
  out_file = os.path.join(expr_name, 'model', 'epoch-200')

  best_epoch, cider = select_best_epoch(log_dir)
  epochs = [best_epoch-10, best_epoch, best_epoch + 10]

  key2vals = {}
  for epoch in epochs:
    model_file = os.path.join(expr_name, 'model', 'epoch-%d'%epoch)
    if not os.path.exists(model_file + '.meta'):
      continue
    name2var = framework.util.graph_ckpt.load_variable_in_ckpt(model_file)
    # print name2var.keys()
    for name in name2var:
      if name not in key2vals:
        key2vals[name] = []
      key2vals[name].append(name2var[name])

  key2val = {}
  for key in key2vals:
    vals = key2vals[key]
    vals = np.array(vals)
    val = np.mean(vals, 0)
    key2val[key] = val

  path_cfg = gen_driver.vevd.gen_dir_struct_info(path_cfg_file)
  model_cfg = gen_driver.vevd.load_and_fill_model_cfg(model_cfg_file, path_cfg)
  m = gen_model.vevd.Model(model_cfg)
  trn_tst_graph = m.build_trn_tst_graph()
  with trn_tst_graph.as_default():
    var_names = [(v.op.name, v.get_shape()) for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)]
    print var_names
    # assign_op, feed_dict = framework.util.graph_ckpt.init_weight_from_singlemodel(model_file, key2val)
    assign_op, feed_dict = tf.contrib.framework.assign_from_values(key2val)

  with tf.Session(graph=trn_tst_graph) as sess:
    sess.run(m.init_op)
    sess.run(assign_op, feed_dict=feed_dict)
    m.saver.save(sess, out_file)


if __name__ == '__main__':
  export_avg_model_weights()

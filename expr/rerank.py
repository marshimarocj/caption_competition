import os
import math

import numpy as np
import mosek

import eval_rank


'''func
'''


'''expr
'''
def graph_match_rerank():
  # root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  pred_files = [
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.1.0', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.1.0', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.flickr30m', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.flickr30m', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5.score', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5.score', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.lstm.max.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.lstm.max.0.5', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'vevd_expr', 'i3d_resnet200.512.512.16.0.5.lstm', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'vevd_expr', 'i3d_resnet200.512.512.16.0.5.lstm', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_rnn_expr', 'i3d_resnet200.300.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'aca_rnn_expr', 'i3d_resnet200.300.0.5', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att.feedforward', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att.feedforward', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.att.flickr30m.feedforward', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.att.flickr30m.feedforward', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.0.1.att.flickr30m.feedforward', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.0.1.att.flickr30m.feedforward', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.npy'),
    os.path.join(root_dir, 'rnnve_expr', 'i3d_i3d_flow_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.npy'),
    os.path.join(root_dir, 'rnnve_expr', 'i3d_i3d_flow_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.npy'),
  ]
  out_files = [
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.1.0', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.1.0', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.flickr30m', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5.score', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5.score', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.lstm.max.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.lstm.max.0.5', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'vevd_expr', 'i3d_resnet200.512.512.16.0.5.lstm', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'vevd_expr', 'i3d_resnet200.512.512.16.0.5.lstm', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_rnn_expr', 'i3d_resnet200.300.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_rnn_expr', 'i3d_resnet200.300.0.5', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att.feedforward', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att.feedforward', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.att.flickr30m.feedforward', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.att.flickr30m.feedforward', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.0.1.att.flickr30m.feedforward', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.0.1.att.flickr30m.feedforward', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.rerank.20.npy'),
    os.path.join(root_dir, 'rnnve_expr', 'i3d_i3d_flow_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    os.path.join(root_dir, 'rnnve_expr', 'i3d_i3d_flow_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.rerank.20.npy'),
  ]

  topk = 20

  for pred_file, out_file in zip(pred_files, out_files):
    predicts = np.load(pred_file)
    predicts = np.exp(predicts)
    num_video, num_caption = predicts.shape

    with mosek.Env() as env:
      with env.Task(0, 0) as task:
        task.appendcons(num_video + num_caption)
        task.appendvars(num_video * num_caption)

        for i, predict in enumerate(predicts):
          idxs = np.argsort(-predict)
          for idx in idxs[:topk]:
            idx_var = i*num_caption + idx
            score = predict[idx]
            task.putcj(idx_var, score)
          for idx in idxs[topk:]:
            idx_var = i*num_caption + idx
            task.putcj(idx_var, 0.)

        for i in range(num_video * num_caption):
          task.putvarbound(i, mosek.boundkey.ra, 0., 1.)

        for i in range(num_video):
          task.putarow(i, range(i*num_caption, (i+1)*num_caption), [1.]*num_caption)
        for i in range(num_caption):
          task.putarow(num_video+i, range(i, num_video*num_caption, num_caption), [1.]*num_video)
        for i in range(num_video + num_caption):
          task.putconbound(i, mosek.boundkey.ra, 0., 1.)

        task.putobjsense(mosek.objsense.maximize)

        task.optimize()

        task.solutionsummary(mosek.streamtype.msg)

        # Get status information about the solution
        solsta = task.getsolsta(mosek.soltype.bas)

        if (solsta == mosek.solsta.optimal or
              solsta == mosek.solsta.near_optimal):
          xx = [0.] * (num_video * num_caption)
          task.getxx(mosek.soltype.bas, # Request the basic solution.
                     xx)
          print("Optimal solution: ")
          xx = np.array(xx)
          xx = xx.reshape((num_video, num_caption))
          np.save(out_file, xx)
        elif (solsta == mosek.solsta.dual_infeas_cer or
            solsta == mosek.solsta.prim_infeas_cer or
            solsta == mosek.solsta.near_dual_infeas_cer or
            solsta == mosek.solsta.near_prim_infeas_cer):
          print("Primal or dual infeasibility certificate found.\n")
        elif solsta == mosek.solsta.unknown:
          print("Unknown solution status")
        else:
          print("Other solution status")


def eval_rerank():
  # root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  pred_files = [
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.1.0', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.1.0', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.gru.max.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.flickr30m', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.flickr30m', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.1.0.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5.score', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.max.0.5.score', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.lstm.max.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.300.150.lstm.max.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'vevd_expr', 'i3d_resnet200.512.512.16.0.5.lstm', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'vevd_expr', 'i3d_resnet200.512.512.16.0.5.lstm', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_rnn_expr', 'i3d_resnet200.300.0.5', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'aca_rnn_expr', 'i3d_resnet200.300.0.5', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att.feedforward', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.300.0.5.att.feedforward', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.att.flickr30m.feedforward', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.att.flickr30m.feedforward', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.0.1.att.flickr30m.feedforward', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'aca_expr', 'i3d_resnet200.500.0.5.0.1.att.flickr30m.feedforward', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct', 'pred', 'val.A.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze', 'pred', 'val.B.npy'),
    # os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze', 'pred', 'val.B.rerank.20.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.npy'),
    # os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.rerank.20.npy'),
    os.path.join(root_dir, 'rnnve_expr', 'i3d_i3d_flow_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.npy'),
    os.path.join(root_dir, 'rnnve_expr', 'i3d_i3d_flow_resnet200.500.250.gru.max.0.5.0.1.flickr30m', 'pred', 'val.A.rerank.20.npy'),
  ]
  label_file = os.path.join(root_dir, 'label', '17.set.2.gt')

  vid2gt = {}
  with open(label_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      vid = int(data[0])
      gt = int(data[1])
      vid2gt[vid] = gt

  predicts = np.load(pred_files[0])
  predicts = np.exp(predicts)
  rerank_predicts = np.load(pred_files[1])
  mir = eval_rank.calc_mir(predicts, vid2gt)
  print mir

  alphas = [.5, .7, .9]
  for alpha in alphas:
    combined_predicts = (1 - alpha) * predicts + alpha * rerank_predicts
    combined_mir = eval_rank.calc_mir(combined_predicts, vid2gt)
    print alpha, combined_mir


def gen_caption_sim_mat():
  # root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  # expr_name = os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m')
  expr_name = os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m')
  # expr_name = os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct')
  caption_embed_files = [
    os.path.join(expr_name, 'pred', 'val.A.embed.npz'),
    os.path.join(expr_name, 'pred', 'val.B.embed.npz'),
  ]
  out_file = os.path.join(expr_name, 'pred', 'sim_AB.npy')

  caption_embeds = []
  for caption_embed_file in caption_embed_files:
    data = np.load(caption_embed_file)
    caption_embed = data['caption_embeds']
    caption_embeds.append(caption_embed)
  sim_AB = np.matmul(caption_embeds[0], caption_embeds[1].T)
  np.save(out_file, sim_AB)


def rwr():
  # root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  # expr_name = os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.500.250.gru.max.0.5.0.1.flickr30m')
  expr_name = os.path.join(root_dir, 'rnnve_expr', 'i3d_resnet200.1000.500.gru.max.0.5.0.1.flickr30m')
  # expr_name = os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct')
  pred_files = [
    [
      os.path.join(expr_name, 'pred', 'val.A.npy'),
      os.path.join(expr_name, 'pred', 'val.A.rerank.20.npy'),
    ],
    [
      os.path.join(expr_name, 'pred', 'val.B.npy'),
      os.path.join(expr_name, 'pred', 'val.B.rerank.20.npy'),
    ]
  ]
  sim_file = os.path.join(expr_name, 'pred', 'sim_AB.npy')
  label_file = os.path.join(root_dir, 'label', '17.set.2.gt')

  vid2gts = [{}, {}]
  with open(label_file) as f:
    for line in f:
      line = line.strip()
      data = line.split(' ')
      vid = int(data[0])
      gts = [int(d) for d in data[1:]]
      vid2gts[0][vid] = gts[0]
      vid2gts[1][vid] = gts[1]

  preds = []
  for pred_file in pred_files:
    pred = (np.load(pred_file[0]) + np.load(pred_file[1])) / 2.
    preds.append(pred)
  preds = np.concatenate(preds, 1) # (num_img, num_txt*2)
  preds = preds.T # (num_txt*2, num_img)
  num = preds.shape[0]

  sim = np.load(sim_file)
  sim = np.maximum(np.zeros(sim.shape), sim)
  row_threshold = -np.sort(-sim, 1)[:, 4:5]
  col_threshold = -np.sort(-sim, 0)[4:5, :]
  row_threshold = np.repeat(row_threshold, num/2, 1)
  col_threshold = np.repeat(col_threshold, num/2, 0)
  threshold = np.maximum(row_threshold, col_threshold)
  sim[sim < threshold] = 0.

  W = np.eye(num)
  W[:num/2, num/2:] = sim
  W[num/2:, :num/2] = sim.T
  row_sum = np.sum(W, 1)
  W /= np.expand_dims(row_sum, 1)

  alphas = [0.1*d for d in range(10)]
  for alpha in alphas:
    A = np.eye(num) - alpha*W # (num_txt*2, num_txt*2)
    b = (1.0 - alpha) * preds # (num_txt*2, num_img)
    x = np.linalg.solve(A, b) # (num_txt*2, num_img)

    pred_A = x[:num/2].T
    pred_B = x[num/2:].T
    mir_A = eval_rank.calc_mir(pred_A, vid2gts[0])
    mir_B = eval_rank.calc_mir(pred_B, vid2gts[1])
    print alpha, mir_A, mir_B


def corr():
  root_dir = '/mnt/data1/jiac/trecvid2018/rank' # neptune
  expr_name = os.path.join(root_dir, 'rnnve_orth_expr', 'i3d_resnet200.512_512_512.250.gru.max.0.5.0.1.flickr30m.freeze.direct')
  caption_embed_files = [
    os.path.join(expr_name, 'pred', 'val.A.embed.npz'),
    os.path.join(expr_name, 'pred', 'val.B.embed.npz'),
  ]
  
  caption_embeds = []
  for caption_embed_file in caption_embed_files:
    data = np.load(caption_embed_file)  
    caption_embed = data['caption_embeds']
    caption_embeds.append(caption_embed)
  caption_embeds = np.concatenate(caption_embeds, 0)
  corr = np.matmul(caption_embeds.T, caption_embeds)
  self_corr = np.power(np.diag(corr), 0.5)
  avg_corr = 0.
  total = 0.
  for i in range(3):
    c = np.abs(corr[i*512:(i+1)*512, (i+1)*512:])
    c /= np.expand_dims(self_corr[i*512:(i+1)*512], 1) * np.expand_dims(self_corr[(i+1)*512:], 0)
    avg_corr += np.sum(c)
    total += c.size

  print avg_corr / total


if __name__ == '__main__':
  # graph_match_rerank()
  eval_rerank()
  # gen_caption_sim_mat()
  # rwr()
  # corr()

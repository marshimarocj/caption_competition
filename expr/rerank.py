import os

import numpy as np
import mosek


'''func
'''


'''expr
'''
def graph_match_rerank():
  # root_dir = '/data1/jiac/trecvid2018/rank' # uranus
  root_dir = '/home/jiac/data/trecvid2018/rank' # gpu9
  pred_file = os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.mean.0.5', 'pred', 'val.B.npy')
  out_file = os.path.join(root_dir, 'ceve_expr', 'i3d_resnet200.300.1_2_3.mean.0.5', 'pred', 'val.B.rerank.20.npy')

  topk = 20

  predicts = np.load(pred_file)
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


if __name__ == '__main__':
  graph_match_rerank()

import os
import json
import sys
import cPickle
sys.path.append('../')

import tornado.ioloop
import tornado.web

import fast_cider


class MainHandler(tornado.web.RequestHandler):
  def initialize(self, cider):
    self.cider = cider

  def post(self):
     data = json.loads(self.request.body)
     pred_captions = [d['pred'] for d in data]
     vids = [d['vid'] for d in data]
     ids = [d['id'] for d in data]

     score, scores = self.cider.compute_cider(pred_captions, vids)

     out = []
     for i, score in enumerate(scores):
        out.append({'id': ids[i], 'score': float(score)})
     self.write(json.dumps({'service': 'cider', 'data': out}))


def prepare_service():
  file = '/mnt/data1/jiac/trecvid2018/rank/annotation/human_caption_dict.pkl' # neptune
  with open(file) as f:
    vid2captions = cPickle.load(f)

  cider = fast_cider.CiderScorer()
  cider.init_refs(vid2captions)

  print 'load complete'

  services = [
    (r'/cider', MainHandler, {
      'cider': cider, 
    }),
  ]

  return services


if __name__ == '__main__':
  services = prepare_service()
  app = tornado.web.Application(services)
  app.listen(8888)
  tornado.ioloop.IOLoop.current().start()
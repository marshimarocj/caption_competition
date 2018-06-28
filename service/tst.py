import os
import urllib
import json

import requests


'''func
'''


'''expr
'''
def tst_post():
  # url = 'localhost:8888'
  # url = '18.216.242.151:8888'
  # url = 'aladdin1.inf.cs.cmu.edu:8888'
  url = '127.0.0.1:8888'

  data = [{'pred': 'a man is playing', 'vid': 999, 'id': 0}]
  r = requests.post('http://' + url + '/cider', json=data)
  data = json.loads(r.text)
  print 'cider:', data

  # data = [{'pred': 'a man is playing', 'gt': ['a man tries to juggle with three water bottles'], 'id': 0}]
  # r = requests.post('http://' + url + '/bleu', json=data)
  # print 'bleu:', json.loads(r.text)

  # r = requests.post('http://' + url + '/rouge', json=data)
  # print 'rouge:', json.loads(r.text)


if __name__ == '__main__':
  tst_post()

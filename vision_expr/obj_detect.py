import os

import tensorflow as tf
from PIL import Image
from object_detection.utils import ops as utils_ops


'''func
'''


'''expr
'''
def tst():
  model_file = '/home/jiac/models/tf/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
  label_map_file = '/home/jiac/toolkit/models/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt'

  NUM_CLASSES = 545

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  label_map = label_map_util.load_labelmap(label_map_file)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)


if __name__ == '__main__':
  tst()

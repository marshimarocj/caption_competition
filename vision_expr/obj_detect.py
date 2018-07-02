import os
import subprocess

import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import GifImagePlugin

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


'''func
'''
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# def run_inference_for_single_image(image, graph):
#   with graph.as_default():
#     with tf.Session() as sess:
#       # Get handles to input and output tensors
#       ops = tf.get_default_graph().get_operations()
#       all_tensor_names = {output.name for op in ops for output in op.outputs}
#       tensor_dict = {}
#       for key in [
#           'num_detections', 'detection_boxes', 'detection_scores',
#           'detection_classes', 'detection_masks'
#       ]:
#         tensor_name = key + ':0'
#         if tensor_name in all_tensor_names:
#           tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
#               tensor_name)

#       image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

#       # Run inference
#       output_dict = sess.run(tensor_dict,
#                              feed_dict={image_tensor: np.expand_dims(image, 0)})

#       # all outputs are float32 numpy arrays, so convert types as appropriate
#       output_dict['num_detections'] = int(output_dict['num_detections'][0])
#       output_dict['detection_classes'] = output_dict[
#           'detection_classes'][0].astype(np.uint8)
#       output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#       output_dict['detection_scores'] = output_dict['detection_scores'][0]
#   return output_dict


def prepare_tensor(graph):
  with graph.as_default():
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  return image_tensor, tensor_dict


def run_inference_for_single_image(image_tensor, tensor_dict, image, sess):
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]

  return output_dict


'''expr
'''
def tst():
  model_file = '/home/jiac/models/tf/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
  label_map_file = '/home/jiac/toolkit/models/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt'
  img_file = '/home/jiac/toolkit/models/research/object_detection/test_images/image1.jpg'
  out_file = '/home/jiac/toolkit/models/research/object_detection/test_images/image1_detect.jpg'

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

  image = Image.open(img_file)
  image_np = load_image_into_numpy_array(image)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  output_dict = run_inference_for_single_image(image_np, detection_graph)

  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8)
  # image_np.save(out_file)
  print image_np.dtype, image_np.shape
  image = Image.fromarray(image_np)
  image.save(out_file)


def extract_imgs_from_gif():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  names = [
    'tumblr_nd746k9J5Q1qbx0eko1_500',
    'tumblr_ni3zr0kGY71tt0tivo1_250',
    'tumblr_npfcfptpJX1u0chl3o1_400',
    'tumblr_nbaio6niSJ1s3ksyfo1_400',
    'tumblr_m931c6H3Tt1qa4llno1_500',
    'tumblr_nfyblj4eZI1rblf33o1_500',
    'tumblr_np1az4Cohq1spi58bo1_400',
  ]
  out_root_dir = os.path.join(root_dir, 'imgs')

  for name in names:
    file = os.path.join(root_dir, 'gif', name + '.gif')
    out_dir = os.path.join(out_root_dir, name)
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)
    cmd = ['convert', '-coalesce', file, os.path.join(out_dir, '%05d.jpg')]
    p = subprocess.Popen(cmd)
    p.wait()


def detect_obj():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  names = [
    'tumblr_nd746k9J5Q1qbx0eko1_500',
    'tumblr_ni3zr0kGY71tt0tivo1_250',
    'tumblr_npfcfptpJX1u0chl3o1_400',
    'tumblr_nbaio6niSJ1s3ksyfo1_400',
    'tumblr_m931c6H3Tt1qa4llno1_500',
    'tumblr_nfyblj4eZI1rblf33o1_500',
    'tumblr_np1az4Cohq1spi58bo1_400',
  ]
  img_root_dir = os.path.join(root_dir, 'imgs')
  out_root_dir = os.path.join(root_dir, 'obj_detect')
  model_file = '/home/jiac/models/tf/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
  label_map_file = '/home/jiac/toolkit/models/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt'

  NUM_CLASSES = 545
  gap = 16

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  image_tensor, tensor_dict = prepare_tensor(detection_graph)

  label_map = label_map_util.load_labelmap(label_map_file)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  with tf.Session(graph=detection_graph) as sess:
    for name in names:
      img_dir = os.path.join(img_root_dir, name)
      img_names = os.listdir(img_dir)
      num = len(img_names)
      out_dir = os.path.join(out_root_dir, name)
      if not os.path.exists(out_dir):
        os.mkdir(out_dir)
      for i in range(num):
        img_file = os.path.join(img_dir, '%05d.jpg'%i)
        image = Image.open(img_file)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(
          image_tensor, tensor_dict, image_np, sess)

        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
        out_file = os.path.join(out_dir, '%05d.jpg'%i)
        image = Image.fromarray(image_np)
        image.save(out_file)


if __name__ == '__main__':
  # tst()
  # extract_imgs_from_gif()
  detect_obj()

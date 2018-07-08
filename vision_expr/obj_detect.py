import os
import argparse
import subprocess

import tensorflow as tf
import numpy as np
import imageio
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import dataset_util


'''func
'''
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


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


def run_inference_for_images(image_tensor, tensor_dict, images, sess):
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: images})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = [int(d) for d in output_dict['num_detections']]
  output_dict['detection_classes'] = [d.astype(np.uint8) for d in output_dict['detection_classes']]
  output_dict['detection_boxes'] = [d for d in output_dict['detection_boxes']]
  output_dict['detection_scores'] = [d for d in output_dict['detection_scores']]

  return output_dict


def non_max_suppression_fast(boxes, overlapThresh):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []
 
  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")
 
  # initialize the list of picked indexes 
  pick = []
 
  # grab the coordinates of the bounding boxes
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
 
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  # idxs = np.argsort(y2)
  idxs = np.array(range(y2.shape[0]), dtype=np.int32)
 
  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
 
    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
 
    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]
 
    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))
 
  # return only the bounding boxes that were picked using the
  # integer data type
  return boxes[pick].astype("int")


'''expr
'''
def tst():
  model_file = '/home/jiac/models/tf/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
  label_map_file = '/home/jiac/toolkit/models/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt'
  img_file = '/home/jiac/toolkit/models/research/object_detection/test_images/image1.jpg'
  out_file = '/home/jiac/toolkit/models/research/object_detection/test_images/image1_detect.jpg'

  NUM_CLASSES = 546

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


def gen_sh_convert_gif_to_mp4():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  out_dir = os.path.join(root_dir, 'mp4')

  split = 8

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      url = line[:pos]
      pos = url.rfind('/')
      name = url[pos+1:]
      name, _ = os.path.splitext(name)
      names.append(name)

  gap = (len(names) + split - 1) / split

  for s in range(split):
    with open('%d.sh'%s, 'w') as fout:
      for name in names:
        file = os.path.join(root_dir, 'gif', name + '.gif')
        out_file = os.path.join(out_dir, name + '.mp4')
        if not os.path.exists(out_dir):
          os.mkdir(out_dir)
        # cmd = ['convert', '-coalesce', file, os.path.join(out_dir, '%05d.jpg')]
        cmd = [
          'ffmpeg', '-i', file, 
          '-movflags', 'faststart', 
          '-pix_fmt', 'yuv420p', 
          '-vf', '"scale=trunc(iw/2)*2:trunc(ih/2)*2"',
          out_file
        ]
        fout.write(' '.join(cmd) + '\n')


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
  # model_file = '/home/jiac/models/tf/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb'
  model_file = '/home/jiac/data/openimage/change_threshold_expr/export/frozen_inference_graph.pb'
  label_map_file = '/home/jiac/toolkit/models/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt'

  NUM_CLASSES = 546
  gap = 16
  max_num = 10

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
      print name
      img_dir = os.path.join(img_root_dir, name)
      img_names = os.listdir(img_dir)
      num = len(img_names)
      out_dir = os.path.join(out_root_dir, name)
      if not os.path.exists(out_dir):
        os.mkdir(out_dir)

      out_boxes = []
      out_classes = []
      out_scores = []
      for i in range(num):
        img_file = os.path.join(img_dir, '%05d.jpg'%i)
        image = Image.open(img_file)
        image = image.convert('RGB')
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(
          image_tensor, tensor_dict, image_np, sess)
        out_boxes.append(output_dict['detection_boxes'][:max_num]) # ymin, xmin, ymax, xmax
        out_classes.append(output_dict['detection_classes'][:max_num])
        out_scores.append(output_dict['detection_scores'][:max_num])

        # vis_util.visualize_boxes_and_labels_on_image_array(
        #   image_np,
        #   output_dict['detection_boxes'],
        #   output_dict['detection_classes'],
        #   output_dict['detection_scores'],
        #   category_index,
        #   min_score_thresh=.05,
        #   max_boxes_to_draw=10,
        #   use_normalized_coordinates=True,
        #   line_thickness=4)
        # out_file = os.path.join(out_dir, '%05d.jpg'%i)
        # image = Image.fromarray(image_np)
        # image.save(out_file)
        # print output_dict['num_detections'], output_dict['detection_classes']

      out_boxes = np.array(out_boxes, dtype=np.float32)
      out_classes = np.array(out_classes, dtype=np.uint8)
      out_scores = np.array(out_scores, dtype=np.float32)
      out_file = out_dir + '.npz'
      np.savez_compressed(out_file, score=out_scores, boxes=out_boxes, classes=out_classes)


def bat_detect_obj():
  # root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  root_dir = '/home/jiac/data/tgif' # gpu8
  gif_dir = os.path.join(root_dir, 'gif')
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  # model_file = '/home/jiac/data/openimage/change_threshold_expr/export/frozen_inference_graph.pb'
  model_file = '/home/jiac/models/tf/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28_threshold/frozen_inference_graph.pb'
  out_dir = os.path.join(root_dir, 'obj_detect')

  NUM_CLASSES = 546
  gap = 16
  split = 4

  parser = argparse.ArgumentParser()
  parser.add_argument('chunk', type=int)
  args = parser.parse_args()
  chunk = args.chunk

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      url = line[:pos]
      pos = url.rfind('/')
      name = url[pos+1:]
      name, _ = os.path.splitext(name)
      names.append(name)
  split_gap = (len(names) + split - 1) / split

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  image_tensor, tensor_dict = prepare_tensor(detection_graph)

  cnt = 0
  with tf.Session(graph=detection_graph) as sess:
    for name in names[chunk*split_gap : (chunk+1)*split_gap]:
      cnt += 1
      if cnt % 100 == 0:
        print cnt

      gif_file = os.path.join(gif_dir, name + '.gif')
      if not os.path.exists(gif_file):
        continue
      out_file = os.path.join(out_dir, name + '.npz')
      if os.path.exists(out_file):
        continue
      try:
        gif = imageio.mimread(gif_file, memtest=False)
      except:
        continue

      out_boxes = []
      out_classes = []
      out_scores = []
      out_frames = []
      for i in range(len(gif)):
        if i % gap < 3:
          if len(gif[i].shape) == 3:
            img = gif[i][:, :, :3]
          else:
            img = gif[i]

          img = Image.fromarray(img)
          img = img.convert('RGB')
          img_np = load_image_into_numpy_array(img)
          output_dict = run_inference_for_single_image(
            image_tensor, tensor_dict, img_np, sess)
          out_boxes.append(output_dict['detection_boxes'])
          out_classes.append(output_dict['detection_classes'])
          out_scores.append(output_dict['detection_scores'])
          out_frames.append(i)
      out_boxes = np.array(out_boxes, dtype=np.float32)
      out_classes = np.array(out_classes, dtype=np.uint8)
      out_scores = np.array(out_scores, dtype=np.float32)
      np.savez_compressed(out_file, 
        scores=out_scores, boxes=out_boxes, classes=out_classes, frames=out_frames)


# def prepare_pseudo_tfrecord():
#   root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
#   names = [
#     'tumblr_nd746k9J5Q1qbx0eko1_500',
#     'tumblr_ni3zr0kGY71tt0tivo1_250',
#     'tumblr_npfcfptpJX1u0chl3o1_400',
#     'tumblr_nbaio6niSJ1s3ksyfo1_400',
#     'tumblr_m931c6H3Tt1qa4llno1_500',
#     'tumblr_nfyblj4eZI1rblf33o1_500',
#     'tumblr_np1az4Cohq1spi58bo1_400',
#   ]
#   img_root_dir = os.path.join(root_dir, 'imgs')
#   out_file = '/home/jiac/data/openimage/pseudo_trn_records/0.record'

#   with tf.python_io.TFRecordWriter(out_file) as writer:
#     for name in names[:1]:
#       img_dir = os.path.join(img_root_dir, name)
#       img_names = os.listdir(img_dir)
#       num = len(img_names)
#       for i in range(num):
#         img_file = os.path.join(img_dir, '%05d.jpg'%i)
#         with tf.gfile.GFile(img_file, 'rb') as fid:
#           encoded_jpg = fid.read()
#         image = Image.open(img_file)
#         w, h = image.size

#         tf_example = tf.train.Example(features=tf.train.Features(feature={
#             'image/height': dataset_util.int64_feature(h),
#             'image/width': dataset_util.int64_feature(w),
#             'image/filename': dataset_util.bytes_feature(img_file),
#             'image/source_id': dataset_util.bytes_feature(img_file),
#             'image/encoded': dataset_util.bytes_feature(encoded_jpg),
#             'image/format': dataset_util.bytes_feature(b"jpg"),
#             'image/object/bbox/xmin': dataset_util.float_list_feature([0.]),
#             'image/object/bbox/xmax': dataset_util.float_list_feature([1.]),
#             'image/object/bbox/ymin': dataset_util.float_list_feature([0.]),
#             'image/object/bbox/ymax': dataset_util.float_list_feature([1.]),
#             'image/object/class/text': dataset_util.bytes_list_feature('Person'),
#             'image/object/class/label': dataset_util.int64_list_feature([1]),
#         }))
#         writer.write(tf_example.SerializeToString())


def prepare_for_matlab():
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
  obj_detect_dir = os.path.join(root_dir, 'obj_detect')

  score_threshold = .05

  for name in names:
    detect_file = os.path.join(obj_detect_dir, name + '.npz')
    data = np.load(detect_file)
    frame_boxes = data['boxes']
    frame_classes = data['classes']
    frame_scores = data['score']

    img_file = os.path.join(root_dir, 'imgs', name, '00000.jpg')
    img = Image.open(img_file)
    img_w, img_h = img.size

    min_size = min(img_w, img_h)/10

    num_frame = frame_scores.shape[0]
    for f in range(0, num_frame, 16):
      all_boxes = []
      for i in range(f, min(f+1, num_frame)):
        sort_idxs = np.argsort(-frame_scores[i])
        valid_idxs = sort_idxs[sort_idxs >= score_threshold]

        boxes = frame_boxes[i][valid_idxs]
        boxes[:, 0] *= img_h
        boxes[:, 1] *= img_w
        boxes[:, 2] *= img_h
        boxes[:, 3] *= img_w
        all_boxes.append(boxes)
      all_boxes = np.concatenate(all_boxes, 0)
      shape = all_boxes.shape
      boxes = non_max_suppression_fast(all_boxes, 0.75)
      print shape, boxes.shape

      out_file = os.path.join(obj_detect_dir, name + '.%d.box'%f)
      with open(out_file, 'w') as fout:
        for box in boxes:
          x = box[1]
          y = box[0]
          w = box[3]-box[1]
          h = box[2]-box[0]
          if w < img_w / 10. or h < img_h / 10.:
            continue
          fout.write('%d %d %d %d\n'%(x, y, w, h))


def bat_prepare_for_matlab():
  root_dir = '/home/jiac/data2/tgif/TGIF-Release/data' # gpu9
  lst_file = os.path.join(root_dir, 'tgif-v1.0.tsv')
  gif_dir = os.path.join(root_dir, 'gif')
  detect_dir = os.path.join(root_dir, 'obj_detect')

  chunk = 0
  score_threshold = .01
  split = 4
  gap = 16

  names = []
  with open(lst_file) as f:
    for line in f:
      line = line.strip()
      pos = line.find(' ')
      url = line[:pos]
      pos = url.rfind('/')
      name = url[pos+1:]
      name, _ = os.path.splitext(name)
      names.append(name)
  split_gap = (len(names) + split - 1) / split

  # for name in names[chunk*split_gap : (chunk+1)*split_gap]:
  for name in names[:5]:
    detect_file = os.path.join(detect_dir, name + '.npz')
    if not os.path.exists(detect_file):
      continue
    print name

    gif_file = os.path.join(gif_dir, name + '.gif')
    gif = imageio.mimread(gif_file, memtest=False)
    img_h = gif[0].shape[0]
    img_w = gif[0].shape[1]

    out_dir = os.path.join(detect_dir, name)
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

    data = np.load(detect_file)
    boxes = data['boxes']
    scores = data['scores']
    num = boxes.shape[0]
    for i in range(0, num, 3):
      all_boxes = []
      all_scores = []
      for j in range(i, min(i+3, num)):
        valid_idxs = scores[j] >= score_threshold
        valid_boxes = boxes[j][valid_idxs]
        valid_boxes[:, 0] *= img_h
        valid_boxes[:, 1] *= img_w
        valid_boxes[:, 2] *= img_h
        valid_boxes[:, 3] *= img_w
        all_boxes.append(valid_boxes)
        all_scores.append(scores[j][valid_idxs])
      all_boxes = np.concatenate(all_boxes, 0)
      all_scores = np.concatenate(all_scores)
      sort_idxs = np.argsort(all_scores)
      all_boxes = all_boxes[sort_idxs]
      suppressed_boxes = non_max_suppression_fast(all_boxes, 0.75)
      print all_boxes.shape, suppressed_boxes.shape

      out_file = os.path.join(out_dir, '%d.box'%(i / 3 * gap))
      with open(out_file, 'w') as fout:
        for box in suppressed_boxes:
          x = box[1]
          y = box[0]
          w = box[3]-box[1]
          h = box[2]-box[0]
          if w < img_w / 10. or h < img_h / 10.:
            continue
          fout.write('%d %d %d %d\n'%(x, y, w, h))


if __name__ == '__main__':
  # tst()
  # extract_imgs_from_gif()
  # gen_sh_convert_gif_to_mp4()
  # detect_obj()
  # bat_detect_obj()
  # prepare_for_matlab()
  bat_prepare_for_matlab()

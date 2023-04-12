import tensorflow as tf
import numpy as np
import urllib.request
import tarfile
import os
from tensorflow.python.platform import gfile
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Inception-v3 모델 다운로드
model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
model_file = 'inception-2015-12-05.tgz'
if not os.path.exists(model_file):
    get_file(model_file, model_url, extract=True)

# Inception-v3 모델 로드
model_dir = os.path.join(os.path.dirname(model_file), 'inception-v3')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 그래프 생성
graph = tf.Graph()
with graph.as_default():
    model_file_path = os.path.join(model_dir, 'classify_image_graph_def.pb')
    with tf.io.gfile.GFile(model_file_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        
class NodeLookup(object):
  def __init__(self, label_lookup_path=None, uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    for line in proto_as_ascii_lines:
      line = line.strip('\n')
      parsed_items = line.split('\t')
      uid = parsed_items[0]
      human_string = parsed_items[1]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


# 이미지 분류 함수 정의
def classify_image(image_url):
    image_data = urllib.request.urlopen(image_url).read()
    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            node_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
            uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
            node_lookup = NodeLookup(node_lookup_path, uid_lookup_path)
            top_k = predictions.argsort()[-5:][::-1]
            results = []
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                results.append({'label': human_string, 'score': float(score)})
    return results

# 이미지 분류 결과 출력 함수 정의
def print_classification_results(results):
    for result in results:
        print('{0:s} (score = {1:.5f})'.format(result['label'], result['score']))

# 사용자 입력 받기
description = input('어떤 사물에 대해서 알고 싶으세요? ')

# 구글 이미지 검색을 이용해 이미지 URL 가져오기
from google_images_search import GoogleImagesSearch
gis = GoogleImagesSearch('AIzaSyBa83KkH8hTv6mqlazkIiDIQVnK4X7Cl5I', 'e6625ab35d99b4405')
gis.search({'q': description})
for image in gis.results():
    print(image.url)
    results = classify_image(image.url)
    print_classification_results(results)
    break  # 가장 상위 결과만 출력하도록 함

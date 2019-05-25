import tensorflow as tf
# https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg
# https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg

cat_in_show = "/mnt/sdb/zhengnianzu/new_project/test/320px-Felis_catus-cat_on_snow.jpg"
williamsburg_bridge = "/mnt/sdb/zhengnianzu/new_project/test/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg"

image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}

# This is an example, just using the cat image.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tf.image.decode_jpeg(image_string).shape

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape
  print(image_shape)
  feature = {
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')

with tf.python_io.TFRecordWriter('images.tfrecords') as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())


raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
image_feature_description = {
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
iter = parsed_image_dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))

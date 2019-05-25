# Using TFRecords and tf.Example
[Using TFRecords and tf.Example](https://www.tensorflow.org/tutorials/load_data/tf_records)

The *TFRecord* format is a simple format for storing a sequence of binary records.

Protocol buffers are a cross-platform, cross-language library for efficient serialization of structured data.

The *tf.Example message* (or protobuf) is a flexible message type that represents a {"string": value} mapping. It is designed for use with TensorFlow and is used throughout the higher-level APIs such as TFX.

## The whole process-Two stage
Encode the message:
```
tf.Example = {"string": tf.train.Feature} 
tf.Example.SerializeToString()
tf.train.Feature(bytes_list, float_list, int64List)
tf.train.BytesList(value=[value])
```
Decode the message:
```
tf.train.Example.FromString(serialized_example)
feature_description = {
    'string': tf.train.Feature
}
tf.parse_single_example(example_proto, feature_description)

```
## Tfrecord file
#### TFRecord files using tf.data.experimental.TFRecordWriter
Write them to a TFRecord file
```
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
```
Reading a TFRecord file
```
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
```
#### TFRecord files using tf.python_io
write a tfrecord file
```
with tf.python_io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)
```
read a tfrecord file

We iterate through the TFRecords in the infile, extract the tf.Example message, and can read/store the values within.
```
record_iterator = tf.python_io.tf_record_iterator(path=filename)

for string_record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(string_record)
  
  print(example)
  
  # Exit after 1 iteration as this is purely demonstrative.
  break
  
print(dict(example.features.feature))
print(example.features.feature['feature3'].float_list.value)
```

### tf.Example
The tf.Example message (or protobuf) is a flexible message type that represents a {"string": value} mapping. 
```
tf.Example                          # a {"string": tf.train.Feature} mapping.
```
```
def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  
  feature = {
      'feature0': _int64_feature(feature0),        # look ahead, meet in the next section
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }
  
  # Create a Features message using tf.train.Example.
  
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()
```
To decode the message use the 
```
tf.train.Example.FromString(serialized_example)       # from string to protomessages
protomessage.SerializeToString                        # All the proto messages can 
                                                        be serialized to binary-string 
```

## TF.train.Features

These files will always be read off of disk in a standardized way and never all at once

Protobuf: a way to serialize data structures, given some schema describing what the data is.

Basically, an Example always contains Features. Features contains a map of strings to Feature. And finally, a Feature contains one of a FloatList, a ByteList or a Int64List.

```
tf.io.TFRecordWriter
attribute:
close()
flush()
write(record)                      # write a string record to file
```
```
tf.train.Features(features={<key:value>,...})                  # A protocolMessage
tf.train.Feature(
bytes_list={}, 
float_list={},
int64_list={},)                   # A protocolMessage
where:
tf.train.Byteslist(value=[value])                 # A protocolMessage, String, byte
tf.train.FloatList(value=[value])                 # float, double
tf.train.Int64list(value=[value])                 # bool, enum, int32, uint32, int64, uint64
```

Shortcut function
```
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```


## tf.FixedLenSequenceFeature
```
tf.FixedLenSequenceFeature(
    shape,
    dtype,
    allow_missing=False,
    default_value=None
)
```
Configuration for parsing a variable-length input feature into a Tensor.

The resulting Tensor of parsing a single SequenceExample or Example has a static shape of \[None] + shape and the specified dtype. The resulting Tensor of parsing a batch_size many Examples has a static shape of \[batch_size, None] + shape and the specified dtype. The entries in the batch from different Examples will be padded with default_value to the maximum length present in the batch.

To treat a sparse input as dense, provide allow_missing=True; otherwise, the parse functions will fail on any examples missing this feature.

## tf.parse_single_sequence_example
```
tf.io.parse_single_sequence_example(
    serialized,
    context_features=None,
    sequence_features=None,
    example_name=None,
    name=None
)
```

## tf.train.data.group_by_window
```
tf.contrib.data.group_by_window(
    key_func,
    reduce_func,
    window_size=None,
    window_size_func=None
)
```


## tf.data.TFRecordDataset and tf.data.experimental.TFRecordWriter
Writing a TFRecord file and then Reading a TFRecord file
```
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
```
```
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
raw_dataset
for raw_record in raw_dataset.take(10):
  print(repr(raw_record))
  
# Create a description of the features.  
feature_description = {
    'feature0': tf.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset 
for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))
```
### TFRecord files using tf.python_io
```
# Write the `tf.Example` observations to the file.
with tf.python_io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)
```
```
record_iterator = tf.python_io.tf_record_iterator(path=filename)

for string_record in record_iterator:
  example = tf.train.Example()
  example.ParseFromString(string_record)
  
  print(example)
  
  # Exit after 1 iteration as this is purely demonstrative.
  break
```





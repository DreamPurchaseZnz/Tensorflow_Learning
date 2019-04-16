# TF EXAMPLE
[Using TFRecords and tf.Example](https://www.tensorflow.org/tutorials/load_data/tf_records)

The processes of getting data into a model can be rather annoying, with a lot of glue code. 

TensorFlow tries to fix this by providing a few ways to feed in data. 
The easiest of these is to use 
```
placeholders
```
which allow you to manually pass in numpy arrays of data.

The second method, my preferred, is to do as much as I possibly can on the graph and to make use of 
```
binary files and input queues
```
Not only does this lighten the amount of code I need to write, removing the need to do any data augmentation or file reading, but the interface is reasonably standard across different kinds of data. It is also conceptually cleaner. 


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
      'feature0': _int64_feature(feature0),
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
## TFRecord files using tf.data
###  tf.contrib.data and tf.data
[Porting your code to tf.data](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/data/README.md)

The tf.contrib.data.Dataset class has been renamed to tf.data.Dataset,
and the tf.contrib.data.Iterator class has been renamed to tf.data.Iterator.
Most code can be ported by removing .contrib from the names of the classes.

### tf.data
```
tf.data                  # provides tools for reading and writing data in tensorflow
```


#### tf.data.TFRecordDataset and tf.data.experimental.TFRecordWriter
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

## To conclude
```
1. Define a record reader function of some kind. This parses the record.
2. Construct a batcher of some kind.
3. Build the rest of your model from these ops.
4. Initialize all the ops.
5. Start queue runners.
6. Run your train loop.
```



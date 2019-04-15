# TensorFlow Data Input : Placeholders, Protobufs & Queues
[Using TFRecords and tf.Example](https://www.tensorflow.org/tutorials/load_data/tf_records)

# Table of contents
1. [Preamble](#preamble)
2. [Placeholders](#Placeholders)
3. [protobuf and binary formats](#protobuf and binary formats)


## tf.contrib.data and tf.data
[Porting your code to tf.data](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/data/README.md)

The tf.contrib.data.Dataset class has been renamed to tf.data.Dataset,
and the tf.contrib.data.Iterator class has been renamed to tf.data.Iterator.
Most code can be ported by removing .contrib from the names of the classes.

## Preamble

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

## Placeholders

You need to have consistent image loading and a scaling pipeline that is consitent at train and test time.
You need to ensure that the input is fast enough.

## protobuf and binary formats

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
protomessage.SerializeToString                        # All the proto messages can be serialized to binary-string 
```
## TFRecord files using tf.data
```

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



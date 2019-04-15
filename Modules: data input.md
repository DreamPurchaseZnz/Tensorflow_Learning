# TensorFlow Data Input : Placeholders, Protobufs & Queues

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

## To conclude
```
1. Define a record reader function of some kind. This parses the record.
2. Construct a batcher of some kind.
3. Build the rest of your model from these ops.
4. Initialize all the ops.
5. Start queue runners.
6. Run your train loop.
```



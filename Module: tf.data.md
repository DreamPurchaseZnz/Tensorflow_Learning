# TF DATA
```
tf.data.Dataset                      # for input pipelines
```
## Classes
```
class Dataset                     # Represents a potentially large set of elements.
class FixedLengthRecordDataset    # A Dataset of fixed-length records from one or more binary files.
class Iterator                    # Represents the state of iterating through a Dataset.
class Options                     # Represents options for tf.data.Dataset.
class TFRecordDataset             # A Dataset comprising records from one or more TFRecord files.
class TextLineDataset             # A Dataset comprising lines from one or more text files.
```

## tf.data.dataset
represents a potentially large set of elements

To start an input pipeline, you must define a source. For example, to construct a Dataset from some tensors in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices(). Alternatively, if your input data are on disk in the recommended TFRecord format, you can construct a tf.data.TFRecordDataset.

Once you have a Dataset object, you can transform it into a new Dataset by chaining method calls on the tf.data.Dataset object. For example, you can apply per-element transformations such as Dataset.map() (to apply a function to each element), and multi-element transformations such as Dataset.batch(). See the documentation for tf.data.Dataset for a complete list of transformations.

```
tf.data.dataset       
```

## Properties
```
output_classes                     # The expected values are tf.Tensor and tf.SparseTensor
output_shapes                      # each componet of an element of this dataset
output_types
```
```
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```
```
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```

### Methods
```
apply(transformation_func)            # Applies a transformation function to this dataset.
```
```
batch(
    batch_size,
    drop_remainder=False)             # Combines consecutive elements of this dataset into batches.
```
```
cache(filename='')                    # Caches the elements in this dataset
concatenate(dataset)                  # Creates a Dataset by concatenating given dataset with this dataset
```
Note that if tensors contains a NumPy array, and eager execution is not enabled, the values will be embedded in the graph as one or more tf.constant operations. For large datasets (> 1 GB), this can waste memory and run into byte limits of graph serialization.
If tensors contains one or more large NumPy arrays, consider the alternative described in this guide.
```
from_tensor_slices(tensors)           # Creates a Dataset whose elements are slices of the given tensors
from_tensors(tensors)                 # Creates a Dataset with a single element, comprising the given tensors
```
```
interleave(                           # Maps map_func across this dataset, and interleaves the results
    map_func,
    cycle_length,                     # controls the number of input elements that are processed concurrently.
    block_length=1,                   # The number of consecutive elements to produce from each input element 
                                        before cycling to another input element.
    num_parallel_calls=None)          
```
```
@staticmethod
list_files(                           # A dataset of all files matching one or more glob patterns.
    file_pattern,
    shuffle=None,
    seed=None
)
```
```
make_initializable_iterator(shared_name=None)    # Creates an Iterator for enumerating 
                                                   The elements of this dataset.
make_one_shot_iterator()                         # A "one-shot" iterator does not currently 
                                                   support re-initialization.
```
```
map(                                             # Maps map_func across the elements of this dataset.
    map_func,
    num_parallel_calls=None
)
```
where Map: Input signature of map_func
```
a = { 1, 2, 3, 4, 5 }
result = a.map(lambda x: ...)

b = { (1, "foo"), (2, "bar"), (3, "baz") }
result = b.map(lambda x_int, y_str: ...)

c = { {"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}, {"a": 3, "b": "baz"} }
result = c.map(lambda d: ...)
```
```
options()                               # Returns the options for this dataset and its inputs.

padded_batch(                           # Combines consecutive elements of this dataset into padded batches.
    batch_size,
    padded_shapes,
    padding_values=None,
    drop_remainder=False
)
```
where padded_batch is designed for the case in which the input elements to be batched may have different shapes, and this transformation will pad each component to the respective shape in padding_shapes, and the padded shapes

The above recipe works for tensors that all have the same size. However, many models (e.g. sequence models) work with input data that can have varying size (e.g. sequences of different lengths). To handle this case, the Dataset.padded_batch() transformation enables you to batch tensors of different shape by specifying one or more dimensions in which they may be padded.
```
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None,))

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
```
The Dataset.padded_batch() transformation allows you to set different padding for each dimension of each component, and it may be variable-length (signified by None in the example above) or constant-length. It is also possible to override the padding value, which defaults to 0.


```
padded_shapes = tf.Dimension(37)       # the component will be padded out to that length in that dimension    
              = tf.Dimension(None)     # the component will be padded out to the maximum length of
                                         all elements in that dimension
```
```
prefetch(buffer_size)                  # Creates a Dataset that prefetches elements from this dataset
```
```
@staticmethod
range(*args)                           # Creates a Dataset of a step-separated range of values.

```
```
reduce(                                # Reduces the input dataset to a single element.
    initial_state,
    reduce_func
)
```
```
repeat(count=None)                     # Repeats this dataset count times.
```
Dataset.repeat() transformation concatenates its arguments without signaling the end of one epoch and the beginning of the next epoch.
```
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```
If you want to receive a signal at the end of each epoch, you can write a training loop that catches the tf.errors.OutOfRangeError at the end of a dataset. At that point you might collect some statistics (e.g. the validation error) for the epoch.
```
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [Perform end-of-epoch calculations here.]
```


```
shuffle(                               # Randomly shuffles the elements of this dataset.
    buffer_size,
    seed=None,
    reshuffle_each_iteration=None
)
```
```
skip(count)                            # Creates a Dataset that skips count elements from this dataset.
take(count)                            # Creates a Dataset with at most count elements from this dataset.
```
```
window(                                # Combines input elements into a dataset of windows.
    size,
    shift=None,                        # The shift argument determines the shift of the window
    stride=1,                          # The stride of the input elements
    drop_remainder=False)             
```
where window
```
tf.data.Dataset.range(7).window(2) produces { {0, 1}, {2, 3}, {4, 5}, {6}} 
tf.data.Dataset.range(7).window(3, 2, 1, True) produces { {0, 1, 2}, {2, 3, 4}, {4, 5, 6}} - 
tf.data.Dataset.range(7).window(3, 1, 2, True) produces { {0, 2, 4}, {1, 3, 5}, {2, 4, 6}}
```
```
@staticmethod
zip(datasets)                  # Creates a Dataset by zipping together the given datasets.
```
       
## tf.data.TFRecordDataset
```
tf.data.Dataset                     # numpy,etc.
tf.data.TFRecordDataset             # comprising records from one or more TFRecord files
__init__(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None)
```
Properties and methods are similar to the tf.data.Dataset.

## tf.data.Iterator
The most common way to consume values from a Dataset is to make an iterator object that provides access to one element of the dataset at a time (for example, by calling Dataset.make_one_shot_iterator()). A tf.data.Iterator provides two operations: Iterator.initializer, which enables you to (re)initialize the iterator's state; and Iterator.get_next(), which returns tf.Tensor objects that correspond to the symbolic next element. Depending on your use case, you might choose a different type of iterator, and the options are outlined below.

```
__init__(
    iterator_resource,    # Creates a new iterator from the given iterator resource.
    initializer,
    output_types,
    output_shapes,
    output_classes
)
```
Note: Most users will not call this initializer directly, and will instead use 
```
Dataset.make_initializable_iterator() 
Dataset.make_one_shot_iterator()                   # iterates through it once
```
### Create an iterator
Once you have built a Dataset to represent your input data, the next step is to create an Iterator to access elements from that dataset. The tf.data API currently supports the following iterators, in increasing level of sophistication:
```
one-shot,
initializable,
reinitializable, and
feedable.
```
#### one-shot
A one-shot iterator is the simplest form of iterator, which only supports iterating once through a dataset, with no need for explicit initialization. One-shot iterators handle almost all of the cases that the existing queue-based input pipelines support, but they do not support parameterization. Using the example of Dataset.range():

```
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  assert i == value
```
#### initializable Iterator
An initializable iterator requires you to run an explicit iterator.initializer operation before using it. In exchange for this inconvenience, it enables you to parameterize the definition of the dataset, using one or more tf.placeholder() tensors that can be fed when you initialize the iterator. Continuing the Dataset.range() example:
```
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```
#### reinitializable Iterator
A reinitializable iterator can be initialized from multiple different Dataset objects. For example, you might have a training input pipeline that uses random perturbations to the input images to improve generalization, and a validation input pipeline that evaluates predictions on unmodified data. These pipelines will typically use different Dataset objects that have the same structure (i.e. the same types and compatible shapes for each component).
```
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```
#### feedable iterator
A feedable iterator can be used together with tf.placeholder to select what Iterator to use in each call to tf.Session.run, via the familiar feed_dict mechanism. It offers the same functionality as a reinitializable iterator, but it does not require you to initialize the iterator from the start of a dataset when you switch between iterators. For example, using the same training and validation example from above, you can use tf.data.Iterator.from_string_handle to define a feedable iterator that allows you to switch between the two datasets:
```
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
while True:
  # Run 200 steps using the training dataset. Note that the training dataset is
  # infinite, and we resume from where we left off in the previous `while` loop
  # iteration.
  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})

  # Run one pass over the validation dataset.
  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})
```

### Properties
```
initializer                # A tf.Operation that should be run to initialize this iterator.
output_classes
output_shapes
output_types
```
### methods
This method allows you to define a "feedable" iterator where you can choose between concrete iterators by feeding a value in a tf.Session.run call. In that case, string_handle would be a tf.placeholder, and you would feed it with the value of tf.data.Iterator.string_handle in each step.
```
@staticmethod
from_string_handle(      # Creates a new, uninitialized Iterator based on the given handle.
    string_handle,
    output_types,
    output_shapes=None,
    output_classes=None
)
```
The returned iterator obtained by from_structure is not bound to a particular dataset, 
and it has no initializer. To initialize the iterator, run the operation returned by Iterator.make_initializer(dataset).
```
@staticmethod
from_structure(        # used to create an iterator that is reusable with many different datasets
    output_types,
    output_shapes=None,
    shared_name=None,
    output_classes=None
)

```
The operation returned by Iterator.get_next() yields the next element of a Dataset when executed, and typically acts as the interface between input pipeline code and your model
```
get_next(name=None)          # a nested structure of tf.Tensors 
                               representing the next element
```
If each element of the dataset has a nested structure, 
the return value of Iterator.get_next() will be one or more tf.Tensor objects in the same nested structure:
```
make_initializer(            #  initializes this iterator on dataset.
    dataset,
    name=None
)
```
```
string_handle(name=None)     # Returns a string-valued tf.Tensor that represents this iterator.
```

```
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset")  # ==> "End of dataset"
```



## [datasets guide](https://www.tensorflow.org/guide/datasets)
A tf.data.Iterator provides the main way to extract elements from a dataset.
       
The simplest iterator is a "one-shot iterator", which is associated with a particular Dataset and iterates through it once       

### Saving iterator state

The tf.contrib.data.make_saveable_from_iterator function creates a SaveableObject from an iterator, which can be used to save and restore the current state of the iterator (and, effectively, the whole input pipeline). A saveable object thus created can be added to tf.train.Saver variables list or the tf.GraphKeys.SAVEABLE_OBJECTS collection for saving and restoring in the same manner as a tf.Variable. Refer to Saving and Restoring for details on how to save and restore variables.
```
# Create saveable object from iterator.
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

# Save the iterator state by adding it to the saveable objects collection.
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tf.train.Saver()

with tf.Session() as sess:

  if should_checkpoint:
    saver.save(path_to_checkpoint)

# Restore the iterator state.
with tf.Session() as sess:
  saver.restore(sess, path_to_checkpoint)
```
### Reading input data
#### Consuming NumPy arrays
If all of your input data fit in memory, the simplest way to create a Dataset from them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices()
```
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```
Note that the above code snippet will embed the features and labels arrays in your TensorFlow graph as tf.constant() operations. This works well for a small dataset, but wastes memory---because the contents of the array will be copied multiple times---and can run into the 2GB limit for the tf.GraphDef protocol buffer.

As an alternative, you can define the Dataset in terms of tf.placeholder() tensors, and feed the NumPy arrays when you initialize an Iterator over the dataset.
```
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```



## [TensorFlow Dataset API tutorial – build high performance data pipelines](https://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/)

The TensorFlow Dataset framework has two main components:
```
Dataset
An associated Iterator
```
The Dataset is basically where the data resides. This data can be loaded in from a number of sources – existing tensors, numpy arrays and numpy files, the TFRecord format and direct from text files. Once you’ve loaded the data into the Dataset object, you can string together various operations to apply to the data, these include operations such as:

```
batch()              # this allows you to consume the data from your TensorFlow Dataset in batches
map()                # this allows you to transform the data using lambda statements applied to each element
zip()                # this allows you to zip together different Dataset objects 
                       into a new Dataset, in a similar way to the Python zip function
filter()             # this allows you to remove problematic data-points in your data-set,
                       again based on some lambda function
repeat()             # this operation restricts the number of times data is consumed 
                       from the Dataset before a tf.errors.OutOfRangeError error is thrown
shuffle()            # this operation shuffles the data in the Dataset
```

### Simple TensorFlow Dataset examples
In the first simple example, we’ll create a dataset out of numpy ranges:
```
x = np.arange(0, 10)
```
We can create a TensorFlow Dataset object straight from a numpy array using from_tensor_slices():
```
# create dataset object from numpy array
dx = tf.data.Dataset.from_tensor_slices(x)
```

The object dx is now a TensorFlow Dataset object. The next step is to create an Iterator that will extract data from this dataset. In the code below, the iterator is created using the method make_one_shot_iterator().  The iterator arising from this method can only be initialized and run once – it can’t be re-initialized. The importance of being able to re-initialize an iterator will be explained more later.

```
# create a one-shot iterator
iterator = dx.make_one_shot_iterator()
# extract an element
next_element = iterator.get_next()
```
After the iterator is created, the next step is to setup a TensorFlow operation which can be called from the training code to extract the next element from the dataset. Finally, the dataset operation can be examined by running the following code:
```
with tf.Session() as sess:
    for i in range(11):
        val = sess.run(next_element)
        print(val)
```
This code will print out integers from 0 to 9 but then throw an OutOfRangeError. This is because the code extracted all the data slices from the dataset and it is now out of range or “empty”.

If we want to repeatedly extract data from a dataset, one way we can do it is to make the dataset re-initializable. We can do that by first adjusting the make_one_shot_iterator() line to the following:

```
iterator = dx.make_initializable_iterator()
```
Then, within the TensorFlow session, the code looks like this:

```
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(15):
        val = sess.run(next_element)
        print(val)
        if i % 9 == 0 and i > 0:
            sess.run(iterator.initializer)
```
Note that the first operation run is the iterator.initializer operation. This is required to get your iterator ready for action and if you don’t do this before running the next_element operation it will throw an error. The final change is the last two lines: this if statement ensures that when we know that the iterator has run out of data (i.e. i == 9), the iterator is re-initialized by the iterator.initializer operation. Running this new code will produce: 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4. No error this time!

There are also other things that can be done to manipulate the dataset and how it can be used. First, the batch function:
```
dx = tf.data.Dataset.from_tensor_slices(x).batch(3)
```
```
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(15):
        val = sess.run(next_element)
        print(val)
        if (i + 1) % (10 // 3) == 0 and i > 0:
            sess.run(iterator.initializer)
```

Will produce an output like:
```
[0 1 2] [3 4 5] [6 7 8] [0 1 2] [3 4 5] [6 7 8]
and so on.
```
Next, we can zip together datasets. This is useful when pairing up input-output training/validation pairs of data (i.e. input images and matching labels for each image). The code below does this:
```
def simple_zip_example():
    x = np.arange(0, 10)
    y = np.arange(1, 11)
    # create dataset objects from the arrays
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)
    # zip the two datasets together
    dcomb = tf.data.Dataset.zip((dx, dy)).batch(3)
    iterator = dcomb.make_initializable_iterator()
    # extract an element
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            if (i + 1) % (10 // 3) == 0 and i > 0:
                sess.run(iterator.initializer)
```
```
dcomb = tf.data.Dataset.zip((dx, dy)).repeat().batch(3)
```
#### TensorFlow Dataset MNIST example

First, we have to load the data from the package and split it into train and validation datasets. This can be performed with the following code:
```
# load the data
digits = load_digits(return_X_y=True)
# split into train and validation sets
train_images = digits[0][:int(len(digits[0]) * 0.8)]
train_labels = digits[1][:int(len(digits[0]) * 0.8)]
valid_images = digits[0][int(len(digits[0]) * 0.8):]
valid_labels = digits[1][int(len(digits[0]) * 0.8):]
```
The load_digits method will extract the data from the relevant location in the scikit-learn package, and the code above splits the first 80% of the data into the training arrays, and the remaining 20% into the validation arrays.
```
# create the training datasets
dx_train = tf.data.Dataset.from_tensor_slices(train_images)
# apply a one-hot transformation to each label for use in the neural network
dy_train = tf.data.Dataset.from_tensor_slices(train_labels).map(lambda z: tf.one_hot(z, 10))
# zip the x and y training data together and shuffle, batch etc.
train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)
```
Finally, the training x and y data is zipped together in the full train_dataset. Chained along together with this zip method is first the shuffle() dataset method. This method randomly shuffles the data, using a buffer of data specified in the argument – 500 in this case. Next, the repeat() method is used, to allow the iterator to continuously extract data from this dataset, finally the data is batched with a batch size of 30.

The same steps are used to create the validation dataset:
```
# do the same operations for the validation set
dx_valid = tf.data.Dataset.from_tensor_slices(valid_images)
dy_valid = tf.data.Dataset.from_tensor_slices(valid_labels).map(lambda z: tf.one_hot(z, 10))
valid_dataset = tf.data.Dataset.zip((dx_valid, dy_valid)).shuffle(500).repeat().batch(30)
```
Now, we want to be able to extract data from either the train_dataset or the valid_dataset seamlessly. This is important, as we don’t want to have to change how data flows through the neural network structure when all we want to do is just change the dataset the model is consuming. To do this, we can use another way of creating the Iterator object – the from_structure() method. This method creates a generic iterator object – all it needs is the data types of the data it will be outputting and the output data size/shape in order to be created. The code below uses this methodology:

```
# create general iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
next_element = iterator.get_next()
```
The second line of the above creates a standard get_next() iterator operation which can be called to extract data from this generic iterator structure. Next, we need some operations which can be called during training or validating to initialize this generic iterator and “point it” to the desired dataset. These are as follows:
```
# make datasets that we can initialize separately, but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
validation_init_op = iterator.make_initializer(valid_dataset)
```
These operations can be run to “switch over” the iterator from one dataset to another. This “switching over” will be demonstrated in code below

Next, the neural network model is created – this is standard TensorFlow usage and in this case I will be utilizing the TensorFlow layers API to create a simple fully connected or dense neural network, with dropout and a first layer of batch normalization to effectively scale the input data. If you’d like to learn some of the basics of TensorFlow, check out my Python TensorFlow tutorial. The TensorFlow model is defined as follows:
```
def nn_model(in_data):
    bn = tf.layers.batch_normalization(in_data)
    fc1 = tf.layers.dense(bn, 50)
    fc2 = tf.layers.dense(fc1, 50)
    fc2 = tf.layers.dropout(fc2)
    fc3 = tf.layers.dense(fc2, 10)
    return fc3
```

To call this model creation function, the code below can be used:
```
# create the neural network model
logits = nn_model(next_element[0])
```
Note that the next_element operation is handled directly in the model – in other words, it doesn’t need to be called explicitly during the training loop as will be seen below. Rather, whenever any of the operations following this point in the graph are called (i.e. the loss operation, the optimization operation etc.) the TensorFlow graph structure will know to run the next_element operation and extract the data from whichever dataset has been initialized into the iterator. The next_element operation, because it is operating on the generic iterator which is defined by the shape of the train_dataset, is a tuple – the first element (\[0]) will contain the MNIST images, while the second element (\[1]) will contain the corresponding labels. Therefore, next_element\[0] will extract the image data batch and send it into the neural network model (nn_model) as the input data.

Next are some standard TensorFlow operations to calculate the loss function, the optimization step and prediction accuracy (again, for more details see this tutorial or this one):

```
# add the optimizer and loss
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)
# get accuracy
prediction = tf.argmax(logits, 1)
equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
init_op = tf.global_variables_initializer()
```

Now we can run the training loop:

```
# run the training
epochs = 600
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(training_init_op)
    for i in range(epochs):
        l, _, acc = sess.run([loss, optimizer, accuracy])
        if i % 50 == 0:
            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
    # now setup the validation run
    valid_iters = 100
    # re-initialize the iterator, but this time with validation data
    sess.run(validation_init_op)
    avg_acc = 0
    for i in range(valid_iters):
        acc = sess.run([accuracy])
        avg_acc += acc[0]
    print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,                           
    (avg_acc / valid_iters) * 100))
```
As can be observed, before the main training loop is entered into, the session executes the training_init_op operation, which initializes the generic iterator to extract data from train_dataset. After running epochs iterations to train the model, we then want to check how the trained model performs on the validation dataset (valid_dataset). To do this, we can simply run the validation_init_op operation in the session to point the generic iterator to valid_dataset. Then we run the accuracy operation as per normal, knowing that the operation will be calculating the model accuracy based on the validation data, rather than the training data. 


## [Data Input Pipeline Performance](https://www.tensorflow.org/guide/performance/datasets)
### Optimizing Performance
As new computing devices (such as GPUs and TPUs) make it possible to train neural networks at an increasingly fast rate, the CPU processing is prone to becoming the bottleneck. The tf.data API provides users with building blocks to design input pipelines that effectively utilize the CPU, optimizing each step of the ETL process.

Pipelining
To perform a training step, you must first extract and transform the training data and then feed it to a model running on an accelerator. However, in a naive synchronous implementation, while the CPU is preparing the data, the accelerator is sitting idle. Conversely, while the accelerator is training the model, the CPU is sitting idle. The training step time is thus the sum of both CPU pre-processing time and the accelerator training time.

Pipelining overlaps the preprocessing and model execution of a training step. While the accelerator is performing training step N, the CPU is preparing the data for step N+1. Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract and transform the data.

Without pipelining, the CPU and the GPU/TPU sit idle much of the time

The tf.data API provides a software pipelining mechanism through the tf.data.Dataset.prefetch transformation, which can be used to decouple the time data is produced from the time it is consumed. In particular, the transformation uses a background thread and an internal buffer to prefetch elements from the input dataset ahead of the time they are requested. Thus, to achieve the pipelining effect illustrated above, you can add prefetch(1) as the final transformation to your dataset pipeline (or prefetch(n) if a single training step consumes n elements).

To apply this change to our running example, change:
```
dataset = dataset.batch(batch_size=FLAGS.batch_size)
return dataset
```
to
```
dataset = dataset.batch(batch_size=FLAGS.batch_size)
dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
return dataset
```
Note that the prefetch transformation will yield benefits any time there is an opportunity to overlap the work of a "producer" with the work of a "consumer." The preceding recommendation is simply the most common application.

### Parallelize Data Transformation

When preparing a batch, input elements may need to be pre-processed. To this end, the tf.data API offers the tf.data.Dataset.map transformation, which applies a user-defined function (for example, parse_fn from the running example) to each element of the input dataset. Because input elements are independent of one another, the pre-processing can be parallelized across multiple CPU cores. To make this possible, the map transformation provides the num_parallel_calls argument to specify the level of parallelism. For example, the following diagram illustrates the effect of setting num_parallel_calls=2 to the map transformation:

Choosing the best value for the num_parallel_calls argument depends on your hardware, characteristics of your training data (such as its size and shape), the cost of your map function, and what other processing is happening on the CPU at the same time; a simple heuristic is to use the number of available CPU cores. For instance, if the machine executing the example above had 4 cores, it would have been more efficient to set num_parallel_calls=4. On the other hand, setting num_parallel_calls to a value much greater than the number of available CPUs can lead to inefficient scheduling, resulting in a slowdown.
```
dataset = dataset.map(map_func=parse_fn)
```
to
```
dataset = dataset.map(map_func=parse_fn, num_parallel_calls=FLAGS.num_parallel_calls)
```









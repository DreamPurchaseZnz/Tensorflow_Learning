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
```
make_initializer(            #  initializes this iterator on dataset.
    dataset,
    name=None
)
```
```
string_handle(name=None)     # Returns a string-valued tf.Tensor that represents this iterator.
```





## [datasets guide](https://www.tensorflow.org/guide/datasets)
A tf.data.Iterator provides the main way to extract elements from a dataset.
       
The simplest iterator is a "one-shot iterator", which is associated with a particular Dataset and iterates through it once       
       
## [TensorFlow Dataset API tutorial – build high performance data pipelines](https://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/)




































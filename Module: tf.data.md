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

A Dataset can be used to represent an input pipeline as a collection of elements (nested structures of tensors) 
and a "logical plan" of transformations that act on those elements.

```
tf.data.dataset       
```
### Properties
```
output_classes                     # The expected values are tf.Tensor and tf.SparseTensor
output_shapes                      # each componet of an element of this dataset
output_types
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
              = tf.Dimension(None)     # the component will be padded out to the maximum length of all elements in that dimension
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


       
       
       
       
       

# Tensor_Transformation
## Casting
Casting tensor data types in your graph
```
tf.string_to_number
tf.to_double
tf.to_float
tf.to_bfloat16
tf.to_int32
tf.to_int64
tf.cast
tf.bitcast
tf.saturate_cast
```
```
cast(
    x,
    dtype,
    name=None)
```
## Shape and Shaping
U can use to determine the shape of tensor and change the shape of a tensor
```
tf.broadcast_dynamic_shape
tf.broadcast_static_shape
tf.shape                      ---> return the shape of a tensor
tf.shape_n
tf.size
tf.rank
tf.reshape
tf.squeeze
tf.expand_dims
tf.meshgrid
```

Removes dimensions of size 1 from the shape of a tensor
```
squeeze(
    input,
    axis=None,             ---> you can remove the specific size 1 demensions by specific axis
    name=None,
    squeeze_dims=None)
```
Insert a dimension of 1 into tensor's shape at dimension index axis of the input shape.start from 0, if negative, count backward from the end.
```
expand_dims(
    input,
    axis=None,             ---> Scalar; Specifies the dimension index at which to expand the shape of input.
                                Must be in the range [-rank(input) - 1, rank(input)]
    name=None,
    dim=None)

```
```
var = tf.Variable([[1,2],[3,4]])
var Out[4]: 
<tf.Variable 'Variable:0' shape=(2, 2) dtype=int32_ref>
var1 = tf.expand_dims(var, [0,2])
ValueError: 'dim' input must be a tensor with a single value for 'ExpandDims'

var1 = tf.expand_dims(var, [0])
var1 Out[7]: 
<tf.Tensor 'ExpandDims_1:0' shape=(1, 2, 2) dtype=int32>
var2 = tf.expand_dims(var1, 3)
var2 Out[9]: 
<tf.Tensor 'ExpandDims_2:0' shape=(1, 2, 2, 1) dtype=int32>
var3 = tf.expand_dims(var2, 0)
var3 Out[11]: 
<tf.Tensor 'ExpandDims_3:0' shape=(1, 1, 2, 2, 1) dtype=int32>
var4 = tf.squeeze(var3, 0)
var4 Out[13]: 
<tf.Tensor 'Squeeze:0' shape=(1, 2, 2, 1) dtype=int32>
```


*tf.resshape*  use a C-order(row-major order) which means to read/write the elements using C like index oder with the last axis index(position not increment) change the fastest, back to the first axis index change slowest. 

thus, when it is called, the tensor values are fetched from tensor and placed into the output array in row major oder. Let's say you 
have a array, for the tensor is not easily to see the value except you launch a session and initial the value
```
a = np.arange(6).reshape((3, 2)) # [0,1,2,3,4,5]
a
Out[65]: 
array([[0, 1],
       [2, 3],
       [4, 5]])
```
You can think of reshaping as first raveling the array (using the given index order), then inserting the elements from the raveled array into the new array using the same kind of index ordering as was used for the raveling.
```
np.reshape(a, (2, 3))
Out[66]: 
array([[0, 1, 2],
       [3, 4, 5]])
np.reshape(np.ravel(a), (2, 3))
Out[67]: 
array([[0, 1, 2],
       [3, 4, 5]])
np.reshape(a, (2, 3), order='F')
Out[68]: 
array([[0, 4, 3],
       [2, 1, 5]])
np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
Out[69]: 
array([[0, 4, 3],
       [2, 1, 5]])
```


## Slicing and Joining
Slice or extract parts of tensor, join multiple tensors together
```
tf.slice
tf.strided_slice
tf.split
tf.tile
tf.pad
tf.concat
tf.stack
tf.parallel_stack
tf.unstack
tf.reverse_sequence
tf.reverse
tf.reverse_v2
tf.transpose
tf.extract_image_patches
tf.space_to_batch_nd
tf.space_to_batch
tf.required_space_to_batch_paddings
tf.batch_to_space_nd
tf.batch_to_space
tf.space_to_depth
tf.depth_to_space
tf.gather
tf.gather_nd
tf.unique_with_counts
tf.scatter_nd
tf.dynamic_partition
tf.dynamic_stitch
tf.boolean_mask
tf.one_hot
tf.sequence_mask
tf.dequantize
tf.quantize_v2
tf.quantized_concat
tf.setdiff1d
```
```
tf.stack(
    values,              ---> Packs the list of tensors in values into a tensor with rank 
                              one higher than each tensor in values
    axis=0,
    name='stack')
```
```
concat(
    values,
    axis,
    name='concat')
```
```
split(
    value,
    num_or_size_splits,
    axis=0,
    num=None,
    name='split')

```
We can assume that there is one occcasion, we need drop some path in the graph. And we can do the following first.
```
var = tf.Variable(tf.random_normal([64, 32, 32, 64]))
var = [var, var]
var
Out[38]: 
[<tf.Variable 'Variable_2:0' shape=(64, 32, 32, 64) dtype=float32_ref>,
 <tf.Variable 'Variable_2:0' shape=(64, 32, 32, 64) dtype=float32_ref>]
var = tf.convert_to_tensor(var)
var
Out[40]: 
<tf.Tensor 'packed:0' shape=(2, 64, 32, 32, 64) dtype=float32>
Out[16]: 
<tf.Variable 'Variable_1:0' shape=(2, 64, 32, 32, 64) dtype=float32_ref>
```
Then we need some mask 
```
num_columns = var.get_shape().as_list()[0]
mask = tf.random_shuffle([True] + [False]*(num_columns-1))
mask
Out[44]: 
<tf.Tensor 'RandomShuffle:0' shape=(2,) dtype=bool>
tf.cast(mask, var.dtype)
Out[45]: 
<tf.Tensor 'Cast:0' shape=(2,) dtype=float32>
mask = tf.cast(mask, var.dtype)
var6 = tf.transpose(tf.multiply(tf.transpose(var), mask))      ---> Brocasting
var6
Out[49]: 
<tf.Tensor 'transpose_1:0' shape=(2, 64, 32, 32, 64) dtype=float32>
```
then we get the *Var6* which we need. besides, we need to multiply the var6 by factor to offset the drop effect.

The following is the decoding part：
```
var1= tf.split(var, num_or_size_splits=1, axis=0)   ---> here, we can see the num has the priority
var1
Out[26]: 
[<tf.Tensor 'split_4:0' shape=(2, 64, 32, 32, 64) dtype=float32>]
var1= tf.split(var, num_or_size_splits=[1,1], axis=0)
var1
Out[28]: 
[<tf.Tensor 'split_5:0' shape=(1, 64, 32, 32, 64) dtype=float32>,
 <tf.Tensor 'split_5:1' shape=(1, 64, 32, 32, 64) dtype=float32>]

var2 = tf.split(var,var.get_shape().as_list()[0],axis=0)
var2
Out[31]: 
[<tf.Tensor 'split_6:0' shape=(1, 64, 32, 32, 64) dtype=float32>,
 <tf.Tensor 'split_6:1' shape=(1, 64, 32, 32, 64) dtype=float32>]
var3 = [tf.squeeze(tensor, 0) for tensor in var2]
var3
Out[33]: 
[<tf.Tensor 'Squeeze_1:0' shape=(64, 32, 32, 64) dtype=float32>,
 <tf.Tensor 'Squeeze_2:0' shape=(64, 32, 32, 64) dtype=float32>]
var4 = tf.concat(var3, axis=-1)
var4
Out[35]: 
<tf.Tensor 'concat:0' shape=(64, 32, 32, 128) dtype=float32>
```
I guess, there will be a more simple method rather than the method above.
for example *tf.reshape* method

## Fake quantization
Operation used to help training for better quantization accuracy
```
tf.fake_quant_with_min_max_args
tf.fake_quant_with_min_max_args_gradient
tf.fake_quant_with_min_max_vars
tf.fake_quant_with_min_max_vars_gradient
tf.fake_quant_with_min_max_vars_per_channel
tf.fake_quant_with_min_max_vars_per_channel_gradient

```


### The following operators are overloaded in the TensorFlow Python API:
```
__neg__ (unary -)
__abs__ (abs())
__invert__ (unary ~)
__add__ (binary +)
__sub__ (binary -)
__mul__ (binary elementwise *)
__div__ (binary / in Python 2)
__floordiv__ (binary // in Python 3)
__truediv__ (binary / in Python 3)
__mod__ (binary %)
__pow__ (binary **)
__and__ (binary &)
__or__ (binary |)
__xor__ (binary ^)
__lt__ (binary <)
__le__ (binary <=)
__gt__ (binary >)
__ge__ (binary >=)
```

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
    name=None
)
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
Brocasting: the next example first extended on the left with singleton dimensions  to become \[1, 1, 1, 15\]
```
a = np.ones((5,))
b = np.ones((5, 28, 28, 1))
(a*b).shape
(5, 28, 28, 5)
b.shape
(5, 28, 28, 1)
a2 = np.reshape(a, [len(a), 1,1,1])
a2.shape
(5, 1, 1, 1)
(a2*b).shape
(5, 28, 28, 1)
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
    name='stack'
)
```
```
concat(
    values,
    axis,
    name='concat'
)
```
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

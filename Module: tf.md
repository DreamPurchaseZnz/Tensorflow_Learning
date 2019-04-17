Module: tf
-----------------------------------------------------------------------------------
```
tf.reshape
tf.placeholder
tf.Variable

tf.tuple
tf.group
tf.random_uniform
tf.reduce_mean
tf.global_norm
tf.IndexedSlices
tf.assign
tf.clip_by_value
tf.transpose
tf.expand_dims                               ---> Insert a dimension of 1  into a tensor'shape
tf.squeeze                                   ---> Removes dimensions of size 1 from the shape of a tensor
tf.tile                                      ---> Construct a tenosr by tiling a given tensor
tf.concat                                    ---> Concatenates tensors along one dimension
tf.minimum                                   ---> Returns the min of x and y (i.e. x < y ? x : y) element-wise.
tf.maximum   
tf.pad
```
### tf.cond

```
cond(
    pred,                                   ---> a scalar determining whether 
                                                 to return the result of true_fn or false_fn
    true_fn=None,                           ---> the callable to be performed if pred is True
    false_fn=None,                          ---> the callable to be performed if pred is False
    strict=False,                           ---> a boolean that enables/disables strict mode
    name=None,                             
    fn1=None,
    fn2=None
)
```
This is a sample code:
```
z = tf.multiply(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))

```



### tf.expand_dims
insert a dimension of 1 into a tensor's shape
```
expand_dims(
    input,                       ---> A tensor
    axis=None,                   ---> A scalar, Specifies the dimension \
                                      index at which to expand the shape of input
    name=None,
    dim=None
)

```
The dimension index *axis* start at zero , 
if you specify a negative number of axis, it is counted bachward from the end
```
# 't' is a tensor of shape [2]
shape(expand_dims(t, 0)) ==> [1, 2]
shape(expand_dims(t, 1)) ==> [2, 1]
shape(expand_dims(t, -1)) ==> [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
```


The task can be discribed as:
f:(batch_size, y_dim)  ---> (batch_size, 1, 1 ,y_dim) ---> (batch_size, img_dim, img_dim ,y_dim)

First method is easier and better

```
f = tf.expand_dims(tf.expand_dims(f, 1), 2)
f = tf.tile(f, [1, img_dim , img_dim , 1])

```
Second method 
```
f=[]
for i in range(y_dim):
  g = tf.tile(f(:,i),[batch_size,img_dim]).reshape(batch_size,1,img_dim,1)
  f.append(g)
f= tf.concat(g,axis = 0)

```


### tf.pad
```
tf.pad(
    tensor,
    paddings,
    mode='CONSTANT',
    name=None,
    constant_values=0
)
```
Pads a tensor.

This operation pads a tensor according to the paddings you specify. paddings is an integer tensor with shape 
```
[n, 2], 
[[df, db]
[df, db]]
```
where n is the rank of tensor. For each dimension D of input, paddings\[D, 0] indicates how many values to add before the contents of tensor in that dimension, and paddings\[D, 1] indicates how many values to add after the contents of tensor in that dimension. If mode is "REFLECT" then both paddings\[D, 0] and paddings\[D, 1] must be no greater than tensor.dim_size(D) - 1. If mode is "SYMMETRIC" then both paddings\[D, 0] and paddings\[D, 1] must be no greater than tensor.dim_size(D).

The padded size of each dimension D of the output is:

paddings\[D, 0] + tensor.dim_size(D) + paddings\[D, 1]

###  tf.shape
```
tf.shape(
    input,
    name=None,
    out_type=tf.dtypes.int32
)
```
Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of input.

### tf.shape Vs a.get_shape()
I see most people confused about tf.shape(tensor) and tensor.get_shape() Let's make it clear:
```
tf.shape
```
tf.shape is used for dynamic shape. If your tensor's shape is changable, use it. An example: a input is an image with changable width and height, we want resize it to half of its size, then we can write something like:
```
new_height = tf.shape(image)[0] / 2
```
```
tensor.get_shape
```
tensor.get_shape is used for fixed shapes, which means the tensor's shape can be deduced in the graph.

Conclusion: tf.shape can be used almost anywhere, but t.get_shape only for shapes can be deduced from graph.

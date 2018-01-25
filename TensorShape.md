# [Tensorshape](https://www.tensorflow.org/api_docs/python/tf/TensorShape)
A TensorShape represents a possibly-partial shape specification for a Tensor. It may be one of the following:

- Fully-known shape: has a known number of dimensions and a known size for each dimension. e.g. TensorShape([16, 256])
- Partially-known shape: has a known number of dimensions, and an unknown size for one or more dimension. e.g. TensorShape([None, 256])
- Unknown shape: has an unknown number of dimensions, and an unknown size in all dimensions. e.g. TensorShape(None)

## Method
```
as_list              ---> Returns a list of integers or None for each dimension.

```
## Necessary
How to change the tensorshape after you get the tensorshape?  
for example:
```
_input=tf.placeholder(
        dtype=tf.float32,
        shape=[None, 1, 2, 1]
    )
print(_input.get_shape())
[Dimension(None) Dimension(1) Dimension(2) Dimension(1)] 

print(_input.get_shape().as_list())
[?, 1, 2, 1]
```
when we want to use the conv2d , we want to scale up to 
```
[Dimension(None) Dimension(1) Dimension(4) Dimension(1)]
```

## Temporary solution
we can just use the broadcasting method.
```
import numpy as np
out_shape = _input.get_shape()
out_shape * np.array([1, 1, 2, 1])
```

If we want a tensorshape like the following:
```
[Dimension(None) Dimension(1) Dimension(5) Dimension(1)]
```
just do a try
```

Error so just specify the batch size
```




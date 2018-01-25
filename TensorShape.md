# [Tensorshape](https://www.tensorflow.org/api_docs/python/tf/TensorShape)
A TensorShape represents a possibly-partial shape specification for a Tensor. It may be one of the following:

- Fully-known shape: has a known number of dimensions and a known size for each dimension. e.g. TensorShape([16, 256])
- Partially-known shape: has a known number of dimensions, and an unknown size for one or more dimension. e.g. TensorShape([None, 256])
- Unknown shape: has an unknown number of dimensions, and an unknown size in all dimensions. e.g. TensorShape(None)

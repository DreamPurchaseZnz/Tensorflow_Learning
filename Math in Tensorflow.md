# MATH
These are all of the methods and classes in the Math subsections of the Tensorflow Python API

## Arithmetic Operators
```
tf.add(x, y, name=None)
tf.subtract(x, y, name=None)
tf.multiply(x, y, name=None)
tf.scalar_mul(scalar, x)
tf.div(x, y, name=None)
tf.divide(x, y, name=None)
tf.truediv(x, y, name=None)
tf.floordiv(x, y, name=None)
tf.realdiv(x, y, name=None)
tf.truncatediv(x, y, name=None)
tf.floor_div(x, y, name=None)
tf.truncatemod(x, y, name=None)
tf.floormod(x, y, name=None)
tf.mod(x, y, name=None)
tf.cross(a, b, name=None)
```
## Basic Math Functions
```
tf.add_n(inputs, name=None)
tf.abs(x, name=None)
tf.negative(x, name=None)
tf.sign(x, name=None)
tf.reciprocal(x, name=None)
tf.square(x, name=None)
tf.round(x, name=None)
tf.sqrt(x, name=None)
tf.rsqrt(x, name=None)
tf.pow(x, y, name=None)
tf.exp(x, name=None)
tf.log(x, name=None)
tf.log1p(x, name=None)
tf.ceil(x, name=None)
tf.floor(x, name=None)
tf.maximum(x, y, name=None)
tf.minimum(x, y, name=None)
tf.cos(x, name=None)
tf.sin(x, name=None)
tf.lbeta(x, name='lbeta')
tf.tan(x, name=None)
tf.acos(x, name=None)
tf.asin(x, name=None)
tf.atan(x, name=None)
tf.lgamma(x, name=None)
tf.digamma(x, name=None)
tf.erf(x, name=None)
tf.erfc(x, name=None)
tf.squared_difference(x, y, name=None)
tf.igamma(a, x, name=None)
tf.igammac(a, x, name=None)
tf.zeta(x, q, name=None)
tf.polygamma(a, x, name=None)
tf.betainc(a, b, x, name=None)
tf.rint(x, name=None)
```
## Matrix Math Functions
```
tf.diag(diagonal, name=None)
tf.diag_part(input, name=None)
tf.trace(x, name=None)
tf.transpose(a, perm=None, name='transpose')
tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32, name=None)
tf.matrix_diag(diagonal, name=None)
tf.matrix_diag_part(input, name=None)
tf.matrix_band_part(input, num_lower, num_upper, name=None)
tf.matrix_set_diag(input, diagonal, name=None)
tf.matrix_transpose(a, name='matrix_transpose')
tf.batch_matmul(x, y, adj_x=None, adj_y=None, name=None)
tf.matrix_determinant(input, name=None)
tf.matrix_inverse(input, adjoint=None, name=None)
tf.cholesky(input, name=None)
tf.cholesky_solve(chol, rhs, name=None)
tf.matrix_solve(matrix, rhs, adjoint=None, name=None)
tf.matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None)
tf.matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None)
tf.self_adjoint_eig(tensor, name=None)
tf.self_adjoint_eigvals(tensor, name=None)
tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)
```
### tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
Multiplies matrix a by matrix b, producing a * b.

The inputs must, following any transpositions, be tensors of rank >= 2 where the inner 2 dimensions specify valid matrix multiplication arguments, and any further outer dimensions match.
### tf.add_n
```
tf.add_n(
    inputs,
    name=None
)
Adds all input tensors element-wise.
```


## Complex Number Functions
```
tf.complex(real, imag, name=None)
tf.complex_abs(x, name=None)
tf.conj(x, name=None)
tf.imag(input, name=None)
tf.real(input, name=None)
```
## Fourier Transform Functions
```
tf.fft(input, name=None)
tf.ifft(input, name=None)
tf.fft2d(input, name=None)
tf.ifft2d(input, name=None)
tf.fft3d(input, name=None)
tf.ifft3d(input, name=None)
```
## Reduction
```
tf.reduce_sum(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.reduce_prod(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.reduce_min(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.reduce_max(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.reduce_all(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.reduce_any(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.reduce_logsumexp(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)
tf.count_nonzero(input_tensor, axis=None, keep_dims=False, dtype=tf.int64, name=None, reduction_indices=None)
tf.accumulate_n(inputs, shape=None, tensor_dtype=None, name=None)
tf.einsum(equation, *inputs)

```
## Scan
```
tf.cumsum(x, axis=0, exclusive=False, reverse=False, name=None)
tf.cumprod(x, axis=0, exclusive=False, reverse=False, name=None)
```

## Segmentation
```
tf.segment_sum(data, segment_ids, name=None)
tf.segment_prod(data, segment_ids, name=None)
tf.segment_min(data, segment_ids, name=None)
tf.segment_max(data, segment_ids, name=None)
tf.segment_mean(data, segment_ids, name=None)
tf.unsorted_segment_sum(data, segment_ids, num_segments, name=None)
tf.sparse_segment_sum(data, indices, segment_ids, name=None)
tf.sparse_segment_mean(data, indices, segment_ids, name=None)
tf.sparse_segment_sqrt_n(data, indices, segment_ids, name=None)

```

## Sequence Comparison and Indexing
```
tf.argmin(input, axis=None, name=None, dimension=None)
tf.argmax(input, axis=None, name=None, dimension=None)
tf.setdiff1d(x, y, index_dtype=tf.int32, name=None)
tf.where(condition, x=None, y=None, name=None)
tf.unique(x, out_idx=None, name=None)
tf.edit_distance(hypothesis, truth, normalize=True, name='edit_distance')
tf.invert_permutation(x, name=None)
```
## Other Functions and Classes
```
tf.mul(x, y, name=None)
tf.neg(x, name=None)
tf.sub(x, y, name=None)
```








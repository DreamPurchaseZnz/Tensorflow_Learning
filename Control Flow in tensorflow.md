# Control Flow

## Control Flow Operations
Control the execution of operations and add conditional dependencies to your graph
```
tf.identity                  ---> copy operation
tf.tuple
tf.group
tf.no_op
tf.count_up_to
tf.cond
tf.case
tf.while_loop
```
```
tf.cond(
    pred,                                 ---> true_fn if pred is true else false_fn
    true_fn=None,
    false_fn=None,
    strict=False,
    name=None,
    fn1=None,
    fn2=None
)


```
## Logical Operators

add logical operator to your graph
```
tf.logical_and
tf.logical_not
tf.logical_or
tf.logical_xor
```
## Comparison Operators
```
tf.equal
tf.not_equal
tf.less
tf.less_equal
tf.greater
tf.greater_equal
tf.where
```
## Debugging Operations
```
tf.is_finite
tf.is_inf
tf.is_nan
tf.verify_tensor_all_finite
tf.check_numerics
tf.add_check_numerics_ops
tf.Assert
tf.Print
```



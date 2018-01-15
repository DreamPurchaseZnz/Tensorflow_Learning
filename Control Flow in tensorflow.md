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
### What's the difference between deep copy and shallow copy
Shallow copy:
> The variables A and B refer to different areas of memory, 
> when B is assigned to A the two variables refer to the same area of memory. 
> Later modifications to the contents of either are instantly reflected in the contents of other, as they share contents.
> cheap,easier
deep copy:
> The variables A and B refer to different areas of memory
> when B is assigned to A the values in the memory area which A points to are copied into the memory area to which B points
> Later modifications to the contents of either remain unique to A or B; the contents are not shared
> expensive

*tf.identity(var)* belongs to shallow copy,i.e. any time you evaluate it, it will grasp the current value of var.
```
tensor = tf.identity(var)
```
deep copy
> create another variable and set its value to the value currently stored in a variable

```
tensor = tf.Variable(<inital>)
tensor = tensor.assign(var)               ---> equal to tf.assign(ref, value)
```
Here is a simple example:
```
import tensorflow as tf
var = tf.Variable(0.9)
var2 = tf.Variable(0.0)
copy_first_variable = var2.assign(var)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(var2))
0.0
sess.run(copy_first_variable)
Out[20]: 
0.89999998
print(sess.run(var2))
0.9
```
```
sess.run(tf.assign(var, 2))
print(sess.run(var))
print(sess.run(var2))

sess.run(tf.assign(var, 2))
Out[45]: 
2.0
print(sess.run(var))
2.0
print(sess.run(var2))
0.9
```

*tf.cond* perform a side effect in one of brancheds, all the operation that your refer to in either branch must execute before the
conditional is evaluated.
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

```
pred = tf.placeholder(tf.bool, shape=[])
x = tf.Variable([1])
def update_x_2():
  with tf.control_dependencies([tf.assign(x, [2])]):
    return tf.identity(x)
  
y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  print(y.eval(feed_dict={pred: False}))  # ==> [1]
  print(y.eval(feed_dict={pred: True}))   # ==> [2]

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



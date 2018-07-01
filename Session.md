# Session and InteractiveSession
The only difference with a regular Session is that an InteractiveSession installs itself as the default session on construction. The methods tf.Tensor.eval and tf.Operation.run will use that session to run ops.

For example:
```
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print(c.eval())
sess.close()
```
Note that a regular session installs itself as default session when it is created with a statement
### Eval method
```
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session():
  # We can also use 'c.eval()' here.
  print(c.eval())
```
### Run method
```
sess = tf.InteractiveSession()
a = tf.constant([10, 20])
b = tf.constant([1.0, 2.0])
v = sess.run(a)                    # array([10, 20])
v = sess.run([a, b])               # [array([10, 20]), array([1., 2.], dtype=float32)]
```


## Session
Most of time, we are confused when we just build a tensor graph or construct a operation. we do not know exactly what the tensor is. so
this post just talking about how we can just uncover the mysterious veil. There are two usual operations. 

```
sess.run                   ---> as function
tensor.eval                ---> as attribute
```

Before explain how to use it, we look at a simple example, first we define some variables,such as labels, var and so on.
of course there are some usual function, like the *tf.one_hot* , which mean transfer the number to *one_hot* coding.
```
var = tf.random_normal(dtype=tf.float32, shape=[10,2])
ones = tf.ones(shape=(10, 1),dtype=tf.int32)
labels = tf.one_hot(indices=ones, depth=2,on_value=1,off_value=0)

zeros = tf.zeros(shape=(10, 1),dtype=tf.int32)
labels_ = tf.one_hot(indices=zeros, depth=2,on_value=1,off_value=0)
```
Use the above function as discribed, we have two preliminaries: first *global_variable_initializer* and then launch a new session.
Then, *tf.eval* is used as an attribute.
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    labels.eval()
    print(labels.eval())
    print(labels_.eval())
```
However the *sess.run* is used as function
```
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(labels)
sess.run()
```

simple example like that
```
import tensorflow as tf
res = tf.one_hot(indices=[0, 3], depth=4)
with tf.Session() as sess:
    print(sess.run(res))

```
```
[[ 1.  0.  0.  0.]
 [ 0.  0.  0.  1.]]
```
## Conv2d_transpose
There is a more complex example.
```
import tensorflow as tf
import numpy as np
def test_conv2d_transpose():
    # input batch shape = (1, 3, 3, 1) -> (batch_size, height, width, channels) - 2x2x1 image in batch of 1
    x = tf.constant(np.array([[
        [[1], [2], [3]],
        [[3], [4], [5]],
        [[4], [5], [6]],
    ]]), tf.float32)
    # shape = (3, 3, 1, 1) -> (height, width, input_channels, output_channels) - 3x3x1 filter
    f = tf.constant(np.array([
        [[[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]]],
        [[[1]], [[1]], [[1]]]
    ]), tf.float32)
    conv = tf.nn.conv2d_transpose(x, f, output_shape=(1, 6, 6, 1), strides=[1, 2, 2, 1], padding='SAME')
    with tf.Session() as session:
        result = session.run(conv)
    return result
test_conv2d_transpose()
```

# Tesing your code
Basicly, we need to test our self-created code. In my practice, I would like to achieve my model in this way
```
1. draw the rough sketch of total structure
2. Tear it apart and test the slice code sequentially
3. Conbine some module and test the whole function.
```
Also this pipeline throw a quetion that if some substantial setting may ignored when you are just testing your code. And there may exists
anthoer method to build your computational graph, like one-shot method
```
1. write the code to have fine details in the begining 
2. sequentially write the following codes and test the code
3. rewrite some modules in your precious codes and continue the step 2
```
It is also neccessary to launch a session to run your code.
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
run(
    fetches,
    feed_dict=None,
    options=None,
    run_metadata=None
)
```
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
If you have a Tensor t, calling t.eval() is equivalent to calling tf.get_default_session().run(t).

You can make a session the default as follows:
```
t = tf.constant(42.0)
sess = tf.Session()
with sess.as_default():   # or `with sess:` to close on exit
    assert sess is tf.get_default_session()
    assert t.eval() == sess.run(t)
```
The most important difference is that you can use sess.run() to fetch the values of many tensors in the same step:
```
t = tf.constant(42.0)
u = tf.constant(37.0)
tu = tf.mul(t, u)
ut = tf.mul(u, t)
with sess.as_default():
   tu.eval()  # runs one step
   ut.eval()  # runs one step
   sess.run([tu, ut])  # evaluates both tensors in a single step
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

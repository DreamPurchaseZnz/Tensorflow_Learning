# Testing module
there are two different method:
```
sess.run
tensor.eval
```
e.g.
```
var = tf.random_normal(dtype=tf.float32, shape=[10,2])
ones = tf.ones(shape=(10, 1),dtype=tf.int32)
labels = tf.one_hot(indices=ones, depth=2,on_value=1,off_value=0)

zeros = tf.zeros(shape=(10, 1),dtype=tf.int32)
labels_ = tf.one_hot(indices=zeros, depth=2,on_value=1,off_value=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    labels.eval()
    print(labels.eval())
    print(labels_.eval())
```

simple example like that
```
import tensorflow as tf
res = tf.one_hot(indices=[0, 3], depth=4)
with tf.Session() as sess:
    print(sess.run(res))

```

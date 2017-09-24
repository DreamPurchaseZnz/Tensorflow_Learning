# Testing module
```
sess.run
tensor.eval
```
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

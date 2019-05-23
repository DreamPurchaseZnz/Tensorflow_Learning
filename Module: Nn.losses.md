# Loss function - tf.nn.losses
---------------------------------------------------------------------------------------------------

```
sigmoid_cross_entropy_with_logits        ---> Computes sigmoid cross entropy given logits.
softmax_cross_entropy_with_logits        ---> Computes softmax cross entropy between logits and labels.
sparse_softmax_cross_entropy_with_logits ---> Computes sparse softmax cross entropy between logits and labels.
weighted_cross_entropy_with_logits       ---> Computes a weighted cross entropy
l2_loss                                  ---> Computes half the L2 norm of a tensor without the sqrt
l2_normalize                             ---> Normalizes along dimension dim using an L2 norm
log_poisson_loss
nce_loss
sampled_softmax_loss
```
## Mathematic formula Vs the tf.nn.softmax

[The math principle of softmax is in the wiki](https://zh.wikipedia.org/zh-cn/Softmax%E5%87%BD%E6%95%B0) 
the following code is implementation: 
```
import math
z = [1,2,3,4]
z_exp = [math.exp(i) for i in z]
print(z_exp)

[2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236]

sum_z_exp = sum(z_exp)
print(sum_z_exp)

84.7910248837216

softmax = [round(i/sum_z_exp,3) for i in z_exp]
print(softmax)

[0.032, 0.087, 0.237, 0.644]
```

Above it is based on principle now  
we can try the tensorflow function to see whether the result is same with the math.
```
import tensorflow as tf
import numpy as np
sess = tf.Session()

a = tf.constant(np.array([1,2,3,4]),dtype=tf.float32) or tf.constant(np.array([1.,2.,3.,4.]))
print(sess.run(a))
print(sess.run(tf.nn.softmax(a)))

```
```
print(sess.run(tf.nn.softmax(a)))
[ 0.0320586   0.08714432  0.23688284  0.64391428]
```
## tf.nn.softmax_cross_entopy Vs the method based on tf.nn.softmax 

```
import tensorflow as tf
import numpy as np
sess = tf.Session()

y_hat = tf.convert_to_tensor(np.array([[0.5, 1.5, 0.1],[2.2, 1.3, 1.7]]))
sess.run(y_hat)
Out[29]: 
array([[ 0.5,  1.5,  0.1],
       [ 2.2,  1.3,  1.7]])
y_hat_softmax = tf.nn.softmax(y_hat)
sess.run(y_hat_softmax)
Out[31]: 
array([[ 0.227863  ,  0.61939586,  0.15274114],
       [ 0.49674623,  0.20196195,  0.30129182]])
y_true = tf.convert_to_tensor(np.array([[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]]))
sess.run(y_true)
Out[33]: 
array([[ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
loss_per_instance_1 = -tf.reduce_sum(y_true * tf.log(y_hat_softmax), reduction_indices=[1])
sess.run(loss_per_instance_1)
Out[34]: 
array([ 0.4790107 ,  1.19967598])
total_loss_1 = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), reduction_indices=[1]))
sess.run(total_loss_1)
Out[35]: 
0.83934333897877944

```

[**softmax_cross_entropy**](https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with)
just can be interpreted as follows:
```
y_hat_softmax = tf.nn.softmax(y_hat)
total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), [1]))
```
```
loss_per_instance_2 = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true)
sess.run(loss_per_instance_2)
Out[37]: 
array([ 0.4790107 ,  1.19967598])

total_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true))
sess.run(total_loss_2)
Out[39]: 
0.83934333897877922

```


## sigmoid
tf.nn.sigmoid_cross_entropy_with_logits
```
sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```

For brevity, let x = logits, z = labels. The logistic loss is
```
  z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
= (1 - z) * x + log(1 + exp(-x))
= x - x * z + log(1 + exp(-x))
```
For x < 0, to avoid overflow in exp(-x), we reformulate the above
```
  x - x * z + log(1 + exp(-x))
= log(exp(x)) - x * z + log(1 + exp(-x))
= - x * z + log(1 + exp(x))
```
Hence, to ensure stability and avoid overflow, the implementation uses this equivalent formulation
```
max(x, 0) - x * z + log(1 + exp(-abs(x)))
logits and labels must have the same type and shape.
```

# Module: tf.losses
Losses operation for use in neural networks
Fuctions:
```
absolute_difference
add_loss
compute_weighted_loss
cosine_distance
get_losses
get_regularization_loss
get_regularization_losses
get_total_loss
hinge_loss
huber_loss
log_loss
mean_pairwise_squared_error
mean_squared_error                                            ---> Adds a Sum-of-Squares loss to the training procedure.
sigmoid_cross_entropy                                         ---> Add sigmoid cross entropy
softmax_cross_entropy                                         ---> Creates a cross-entropy loss 
sparse_softmax_cross_entropy                                  ---> Cross-entropy loss 
```

# Module:tf.nn
Neural network support

# Functions

### Pooling ,CNN and Normalization 
```
all_candidate_sampler
atrous_conv2d                            ---> Atrous convolution(a.k.a. convolution with holes or dilated convolution)
atrous_conv2d_transpose
avg_pool                                 ---> Performs the average pooling on the input
avg_pool3d
fractional_avg_pool                      ---> Performs fractional average pooling on the input.
fractional_max_pool                      ---> Performs fractional max pooling on the input.
fused_batch_norm                         --->  Batch normalization.
batch_normalization                      ---> Batch normalization.
conv1d                                   ---> Computes a 1-D convolution given 3-D input and filter tensors.
conv2d                                   ---> Computes a 2-D convolution given 4-D input and filter tensors.
conv2d_backprop_filter                   
conv2d_backprop_input                   
conv2d_transpose                         ---> The transpose of conv2d.
conv3d                                   ---> Computes a 3-D convolution given 5-D input and filter tensors.
conv3d_backprop_filter_v2
conv3d_transpose
convolution                              ---> Computes sums of N-D convolutions (actually cross-correlation).
depthwise_conv2d                         ---> Depthwise 2-D convolution, applies a different filter to each input channel 
depthwise_conv2d_native                 
depthwise_conv2d_native_backprop_filter
depthwise_conv2d_native_backprop_input
local_response_normalization             ---> Local Response Normalization.
max_pool                                 ---> Performs the max pooling on the input.
max_pool3d                               ---> Performs 3D max pooling on the input.
max_pool_with_argmax
moments                                  ---> Calculate the mean and variance of x.
normalize_moments
pool                                     ---> Performs an N-D pooling operation.
quantized_avg_pool                       ---> Produces the average pool of the input tensor for quantized types.
quantized_conv2d
quantized_max_pool
raw_rnn
separable_conv2d
```
### Activation function 
```
xw_plus_b                                --->  Computes matmul(x, weights) + biases.
relu_layer                               ---> Computes Relu(x * weight + biases).
tanh                                     ---> Computes hyperbolic tangent of x element-wise.
crelu                                    ---> Computes Concatenated ReLU.
elu                                      ---> Computes exponential linear: exp(features) - 1 if < 0, features otherwise.
relu                                     ---> Computes rectified linear: max(features, 0).
quantized_relu_x                         ---> Computes Quantized Rectified Linear X: min(max(features, 0), max_value)
relu6                                    ---> Computes Rectified Linear 6: min(max(features, 0), 6).
sigmoid                                  ---> Computes sigmoid of x element-wise.
softmax                                  ---> Computes softmax activations.
log_softmax                              ---> Computes log softmax activations.
softplus                                 ---> Computes softplus: log(exp(features) + 1).
softsign                                 ---> Computes softsign: features / (abs(features) + 1).
dropout                                  ---> Computes dropout
```

### RNN AND EMBEDDING AND SAMPLER 
```
static_bidirectional_rnn
static_rnn
static_state_saving_rnn
sufficient_statistics
top_k
uniform_candidate_sampler
weighted_moments                         ---> Returns the frequency-weighted mean and variance of x.
with_space_to_batch
zero_fraction
ctc_beam_search_decoder
ctc_greedy_decoder
ctc_loss
bias_add
bidirectional_dynamic_rnn                
compute_accidental_hits
dilation2d
dynamic_rnn
embedding_lookup
embedding_lookup_sparse
erosion2d
fixed_unigram_candidate_sampler
in_top_k
learned_unigram_candidate_sampler
log_uniform_candidate_sampler
lrn
```
### Loss function
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
### Example
[The math principle is](https://zh.wikipedia.org/zh-cn/Softmax%E5%87%BD%E6%95%B0) :
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

Above it is based on principle now  we can try the tensorflow function to see whether the result is same with the math.
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

[**softmax_cross_entropy**](https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with) equal to:
```
y_hat_softmax = tf.nn.softmax(y_hat)
total_loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_hat_softmax), [1]))

```
we can verify it through a Experiment
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
```
loss_per_instance_2 = tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y_true)
sess.run(loss_per_instance_2)
Out[37]: 
array([ 0.4790107 ,  1.19967598])
total_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, y_true))
sess.run(total_loss_2)

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

## Module: tf.losses
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


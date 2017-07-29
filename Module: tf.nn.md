# Module:tf.nn
Neural network support

# Functions

* POOLING AND CONV AND NORMALIZATION 
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
* ACTIVATION FUNCTION 
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
* LOSS FUNCTION 
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
* RNN AND EMBEDDING AND SAMPLER 
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

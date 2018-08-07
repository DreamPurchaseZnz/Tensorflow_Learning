# Module:tf.nn
Neural network support

```
all_candidate_sampler
atrous_conv2d                            ---> Atrous convolution
                                             (a.k.a. convolution with holes or
                                             dilated convolution)
atrous_conv2d_transpose
avg_pool                                 ---> Performs the average pooling on the input
avg_pool3d
fractional_avg_pool                      ---> Performs fractional average pooling on the input.
fractional_max_pool                      ---> Performs fractional max pooling on the input.
fused_batch_norm                         --->  Batch normalization.
batch_normalization                      ---> Batch normalization.
conv1d                                   ---> Computes a 1-D convolution given 3-D 
                                              input and filter tensors.
conv2d                                   ---> Computes a 2-D convolution given 4-D 
                                              input and filter tensors.
conv2d_backprop_filter                   
conv2d_backprop_input                   
conv2d_transpose                         ---> The transpose of conv2d.
conv3d                                   ---> Computes a 3-D convolution given 5-D 
                                              input and filter tensors.
conv3d_backprop_filter_v2
conv3d_transpose
convolution                              ---> Computes sums of N-D convolutions 
                                             (actually cross-correlation).
depthwise_conv2d                         ---> Depthwise 2-D convolution, applies a 
                                              different filter to each input channel 
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
quantized_avg_pool                       ---> Produces the average pool of the 
                                              input tensor for quantized types.
quantized_conv2d
quantized_max_pool
raw_rnn
separable_conv2d
```
### tf.nn.conv2d
Compute a 2D convolution given 4-D input and filter tensors
```
conv2d(
    input,                               ---> [batch_size, in_height, in_width, in_channal] 
    filter,                              ---> [filter_height, filter_width, in_channal, out_channal]
    strides,                             ---> A list of ints
    padding,                             ---> A string from: "SAME", "VALID"
    use_cudnn_on_gpu=None,
    data_format=None,                    ---> An optional string from:"NHWC"
                                              ([batch, height, width, channels]),"NCHW" 
    name=None
)
```
### tf.nn.avg_pool
perform the average pooling on the input
```
avg_pool(
    value,                              ---> A 4-D tensor of shape:[batch, height, width, channels]
    ksize,                              ---> A list of ints that has length >=4; 
                                             the size of the window for each dimension of input tensor
    strides,                            ---> A list of ints that has length >=4;
                                             the stride of the sliding 
                                             window for each dimension of input tensor
    padding,                            ---> A string, either "SAME" or "VALID "
    data_format='NHWC',                 ---> A string
    name=None
)

```

### tf.nn.dropout
compute dropout

With probability keep_prob, outputs will be scaled up by 1/ keep_prob, otherwise output 0, this scaling is so that the expect sum
is unchanged.
```
dropout(
    x,                                 ---> A tensor
    keep_prob,                         ---> A scalar Tensor, 
                                            the probability that each element is kept
    noise_shape=None,                  
    seed=None,
    name=None
)


```
### tf.nn.conv2d_transpose
```
conv2d_transpose(
    value,                           ---> A 4-D Tensor of type float and shape [batch, heigh, width, in_channels] for NHWC
    filter,                          ---> shape [heigh, width, out_channels, in_channels],
                                          filters' in channels must match that of the value
    output_shape,                    ---> representing the output shape of the deconvolution op
    strides,
    padding='SAME',
    data_format='NHWC',
    name=None
)
```
here is a practical code, where is wrong ?
```
def deconv2d(_input, kernel_size=3, out_shape=None,
             strides=None, padding='SAME'):
    """out_shape should be spacial size"""
    if strides is None:
        strides = [1, 1, 2, 1]
    n, h, w, in_features = _input.get_shape().as_list()
    filters = weight_variable_msra(
        [1, kernel_size, in_features, in_features],                 ---> here out_filters = in_features
        name='kernel')
    if out_shape is None:
        out_shape = tf.stack([n, 1, w * 2, in_features//2])         ---> by contrast, this indicates the out_filters is in_features//2
    else:
        out_shape = tf.stack([n, 1, out_shape[1], in_features//2])
    return tf.nn.conv2d_transpose(_input, filters, out_shape, strides, padding=padding)

```
The graph can be established, however, when you use optimizer to optimize the function that consists of the *deconv2d*, it will 
raise excertion that 

```
During handling of the above exception, another exception occurred:

'gradients/decoder/decoder_section/block_7/up_block/conv2d_transpose_grad/Conv2D'
(op: 'Conv2D') with input shapes: [64,1,15,64], [1,3,128,128].

During handling of the above exception, another exception occurred:

ValueError: Dimensions must be equal, but are 64 and 128 for 'gradients/decoder/decoder_section/block_7/up_block/conv2d_transpose_grad/Conv2D'
(op: 'Conv2D') with input shapes: [64,1,15,64], [1,3,128,128].
```
So you cannot tell why it was wrong at first glance.


# Activation function 
---------------------------------------------------------------------------------------------------

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



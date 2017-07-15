# Tensorflow 
--------------------------------------------------------------------------------------------------------------
## Higher level ops for building neural network layers
```
tf.contrib.layers.avg_pool2d 
tf.contrib.layers.max_pool2d
tf.contrib.layers.batch_norm 
tf.contrib.layers.convolution2d                --->  Adds an N-D convolution followed by an optional batch_norm layer
tf.contrib.layers.conv2d_in_plane              --->  Performs the same in-plane convolution to each channel independently
tf.contrib.layers.convolution2d_in_plane       --->  Equal to conv2d in plane
tf.nn.conv2d_transpose
tf.contrib.layers.convolution2d_transpose      --->  Equal to conv2d_transpose, the opposite operations against conv2d
tf.nn.dropout
tf.contrib.layers.flatten
tf.contrib.layers.fully_connected
tf.contrib.layers.layer_norm
tf.contrib.layers.linear
tf.contrib.layers.one_hot_encoding
tf.nn.relu
tf.nn.relu6                                    --->  Computes Rectified Linear 6: min(max(features, 0), 6).features belong to [0,6]
tf.contrib.layers.repeat                       --->  Applies the same layer with the same arguments repeated
tf.nn.separable_conv2d                         --->  2-D convolution with separable filters
tf.contrib.layers.separable_convolution2d      --->  Breifly written
tf.nn.softmax
tf.stack                                       --->  Stacks a list of rank-R tensors into one rank-(R+1) tensor
tf.contrib.layers.unit_norm                    --->  Normalizes the given input across the specified dimension to unit length
tf.contrib.layers.embed_sequence
tf.contrib.layers.safe_embedding_lookup_sparse --->  Vacabulary aspect
```
## Regularizers
Regularizers help prevent overfitting.
```
tf.contrib.layers.apply_regularization         --->  Returns the summed penalty by applying regularizer to the weights_list.it is less                                                      helpful for biases.
tf.contrib.layers.l1_regularizer               --->  Apply L1 regularization to weights
tf.contrib.layers.l2_regularizer               --->  Apply L2 regularization to weights
tf.contrib.layers.sum_regularizer              --->  Returns a function that applies the sum of multiple regularizers
```

## Initializers
Initializers is used for initializing variables.When initializing a deep network, it is in principle advantageous to keep the scale of the input variance constant, so it does not explode or diminish by reaching the final layer.
```
tf.contrib.layers.xavier_initializer               --->  Returns an initializer performing "Xavier" initialization for weights.
tf.contrib.layers.xavier_initializer_conv2d        --->  Same as xavier initializer
tf.contrib.layers.variance_scaling_initializer     --->  Returns an initializer without scaling variance(unit variance)
```
## Optimization
Optimize weights given a loss
```
tf.contrib.layers.optimize_loss
```

## Summaries
Helper functions to summarize specific variables or ops
```
tf.contrib.layers.summarize_activation        --->  Adds useful summaries specific to the activation
tf.contrib.layers.summarize_tensor            --->  Adds a summary op for tensor. The type of summary depends on the shape of tensor,
                                                    scalars->scalar_summary ;All other tensors, histogram_summary is used
tf.contrib.layers.summarize_tensors           --->  Summarize a set of tensors
tf.contrib.layers.summarize_collection        --->  Summarize a graph collection of tensors, possibly filtered by name
```



-------------------------------------------------------------------------------------------------------
## Graph collection
Graph collection is used for collect parameters from specific graph
```
tf.add_to_collection(name, value)
tf.get_collection(key, scope=None)   
tf.get_collection_ref(key)
```
About Key is the GraphKeys class which contains many standard names for collections. You can use various preset names to collect and retrive values with a graph.Standard keys are defined as following:
```
key = tf.GraphKeys.TRAINABLE_VARIABLES
```
```
GLOBAL_VARIABLES                   --->  Shared across distributed environment
LOCAL_VARIABLES                    --->  Variable objects that are local to each machine 
MODEL_VARIABLES                    --->  Model for inference
----------------------------------------------------------------------------------------------
Above use tf.contrib.framework to add to this collection
----------------------------------------------------------------------------------------------
TRAINABLE_VARIABLES                --->  Variable that will be trained by an optimzer
SUMMARIES                          --->  Attach to Graph
QUEUE_RUNNERS: 
MOVING_AVERAGE_VARIABLES: 
REGULARIZATION_LOSSES: 
WEIGHTS:
BIASES:
ACTIVATIONS: 
```




## Use Tensorboard:

In the command windows
```
tensorboard --logdir=Path to your folder where your file is.
```
if you can not open http://0.0.0.0:6006, you can try  **http:localhost:6006**







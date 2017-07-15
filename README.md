Tensorflow
---------------------------------------------------------------
A TensorFlow computation, represented as a dataflow graph.A Graph contains a set of tf.Operation objects, which represent units of computation; and tf.Tensor objects, which represent the units of data that flow between operations.

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
                             ABOVE THE FUNCTION LAYERS USED FOR GRAPH BUILDING  

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

----------------------------------------------------------------------------------------------------------
                               SINCE WE HAVE TRAINABLE VARIABLE WE CAN USE IT TO OPTIMIZE LOSS
----------------------------------------------------------------------------------------------------------

## Optimizers
The optimizers provides methods to compute gradients for a loss and apply gradient to variables
```
tf.train.Optimizer
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOptimizer
tf.train.AdagradOptimizer
tf.train.AdagradDAOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.ProximalGradientDescentOptimizer
tf.train.ProximalAdagradOptimizer
tf.train.RMSPropOptimizer
```
Above Optimizers is class type , which have several method
```
__init__             ---> Construct a new optimizer
apply_gradients      
compute_gradients
minimize             ---> Add operation to minimize loss by updating var_list
```
Maybe, Gradients are just unreasonable,so it comes to gradient clipping  
## Gradient Clipping
Gradient clipping ,several operations provided by tensorflow, is used to add clipping function to your graph.Those method can particularly useful for exploding or vanishing gradient.
```
tf.clip_by_value           --->  Clips tensor to specified min and max
tf.clip_by_norm            --->  Clips tensor values to a maximum L2-norm     t= t * clip_norm / l2norm(t)
tf.clip_by_average_norm    --->  Clips tensor values to a maximum average L2-norm
tf.clip_by_global_norm     --->  Clips values of multiple tensors by the ratio of the sum of their norms.
tf.global_norm             --->  Computes the global norm of multiple tensors.
```
Global norm can be written like this:
```
global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
```
## Control dependences & tf.Graph
Once talking about control dependences ,we are entering into the most complex domain -class **tf.Graph**.Let's just briefly introduce it:
```
Properties
  building_function
  finalized
  graph_def_versions
  seed
  version
  
Methods
  __init__
  add_to_collection
  clear_collection                   --->  Clears all values in a collection.
  container                          --->  Returns a context manager that specifies the resource container to use.
  control_dependencies               --->  Returns a context manager that specifies control dependencies.See examples in tensorflow
  device                             --->  Returns a context manager that specifies the default device to use
  finalize                           --->  Finalizes this graph, making it read-only.
  get_all_collection_keys            --->  Returns a list of collections used in this graph
  get_collection                     --->  Returns a list of values in the collection with the given name
  get_collection_ref
  get_name_scope                     --->  Returns the current name scope
  get_operation_by_name              --->  Returns the Operation with the given name
  get_operations                     --->  Return the list of operations in the graph
  get_tensor_by_name
  gradient_override_map
  is_feedable
  is_fetchable
  name_scope                         --->  Returns a context manager that creates hierarchical names for operations.
  prevent_feeding                    --->  Marks the given tensor as unfeedable in this graph
  prevent_fetching                   --->  Marks the given op as unfetchable in this graph
  unique_name
```
As for control dependence we can explain it as following example:
```
with g.control_dependencies([a, b]):
  # Ops constructed here run after `a` and `b`.
  with g.control_dependencies([c, d]):
    # Ops constructed here run after `a`, `b`, `c`, and `d`.

```
In this case,a new Operation will have control dependencies on the union of control_inputs from all active contexts
So What is the point of doing this? obviously, it's used to control computation order. for example ,only you updata gradient can you clip it.  

---------------------------------------------------------------------------------------------------
## Summary Operations
Tensor summaries for exporting information about a model. There are some functions can be used:
```
audio                    --->  Outputs a Summary protocol buffer with audio(Protocol buffers are Google's language-neutral, platform-                                  neutral, extensible mechanism for serializing structured data )
get_summary_description  --->  When a Summary op is instantiated, a SummaryDescription of associated metadata is stored in its NodeDef
histogram                --->  Adding a histogram summary makes it possible to visualize your data's distribution in TensorBoard
image                    --->  Adding a image summary
merge                    --->  Merges summaries
merge-all                --->  Merges all summaries collected in the default graph-key=tf.GraphKeys.SUMMARIES
scalar                   --->  A single scalar value.
tensor_summary           
text                     --->  textual data
```
Then, you can just run the merged summary op, which will generate a serialized Summary protobuf object with all of your summary data at a given step. Finally, to write this summary data to disk, pass the summary protobuf to： 
```
tf.summary.FileWriter
```

Now that you've modified your graph and have a FileWriter, you're ready to start running your network! If you want, you could run the merged summary op every single step, and record a ton of training data. That's likely to be more data than you need, though. Instead, consider running the merged summary op every n steps.

## Tensorboard:
The computations you'll use TensorFlow for - like training a massive deep neural network - can be complex and confusing. To make it easier to understand, debug, and optimize TensorFlow programs, we've included a suite of visualization tools called TensorBoard.

Launching TensorBoard-in the command windows
```
tensorboard --logdir=Path to your folder where your file is.
```
if you can not open http://0.0.0.0:6006, you can try  **http:localhost:6006**

## tf.train.Saver
The Saver class adds ops to save and restore checkpoints,which is a binary files in a propriotary format that map variable names to tensor values.
```
Properties:
  last_checkpoints                      --->  A list of checkpoint filenames, sorted from oldest to newest
  
```
Methods as following:
```
__init__
as_saver_def                           --->  Generates a SaverDef representation of this save
build                                  --->  Proto
export_meta_graph
from_proto
recover_last_checkpoints               --->  Recovers the internal saver state after a crash.
restore                                --->  Restore previously saved variables ,It requires a session in which the graph was launched.                                              The variables to restore do not have to have been initialized, as restoring is itself a                                                way to initialize variables.
save                                   --->  Save variables,it requires a session in which graph was launched and variables was                                                      initialized
set_last_checkpoints_with_time         --->  Sets the list of old checkpoint filenames and timestamps
to_proto

```


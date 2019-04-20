# RNN-recurrent neural networks
---------------------------------------------------------------------------------------------------
## Core RNN Cell wrappers (RNNCells that wrap other RNNCells)

RNN cells:
```
tf.contrib.rnn.BasicRNNCell                       # [batch_size, input_size], Need to feed step by step
tf.contrib.rnn.GRUBlockCellV2                     # Input:(Time, batch, features)
tf.contrib.rnn.LSTMBlockCell                      # Input:(Time, batch, features)
                                                    Call: need state
tf.contrib.rnn.LSTMBlockFusedCell                 # Input:(Time, batch, features)
                                                    Call: only inputs
tf.contrib.rnn.TimeReversedFusedRNN               # Input:(Time, batch, features)
```

Cell concatenation:
```
tf.nn.dynamic_rnn
tf.contrib.rnn.stack_bidirectional_dynamic_rnn    # stack bidirectional rnn
                                                    The combined forward and backward
                                                    layer outputs are used as input of the next layer
tf.nn.rnn_cell.MultiRNNCell                       # stack RNNcell
```

### tf.nn.rnn_cell.BasicRNNCell
```
tf.contrib.rnn.BasicRNNCell
tf.nn.rnn_cell.BasicRNNCell
```
The most basic RNN cell.

Note that this cell is not optimized for performance. Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better performance on GPU.
```
__init__(
    num_units,               #  int, The number of units in the RNN cell.
    activation=None,         #  Nonlinearity to use. Default: tanh
    reuse=None,
    name=None,
    dtype=None,
    **kwargs
)
```
Properties
```
activity_regularizer
dtype
graph
input
input_mask
input_shape
losses
name
non_trainable_variables
non_trainable_weights
output_mask
output_shape
```
Methods:

Run this RNN cell on inputs, starting from the given state
```
__call__(
    inputs,            # 2-D tensor with shape [batch_size, input_size]       
    state,             # 2-D Tensor with shape [batch_size, self.state_size]
    scope=None,
    *args,
    **kwargs
)

apply(
    inputs,            # Apply the layer on a input
    *args,
    **kwargs
)

build(
    instance,
    input_shape
)

compute_mask(
    inputs,
    mask=None
)

compute_output_shape(input_shape)

count_params()

from_config(
    cls,
    config
)

get_config()

get_initial_state(
    inputs=None,
    batch_size=None,
    dtype=None
)

zero_state(
    batch_size,
    dtype
)

set_weights(weights)
get_weights()
get_updates_for(inputs)
get_output_shape_at(node_index)
get_output_mask_at(node_index)
```

### tf.nn.dynamic_rnn
```
tf.nn.dynamic_rnn(
    cell,                      # An instance of RNNCell.
    inputs,                    # [batch_size, max_time, ...]
    sequence_length=None,
    initial_state=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,          #  The shape format of the inputs and outputs Tensors
    scope=None
)
```
```
# create a BasicRNNCell
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

# defining initial state
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
```

```
# create 2 LSTMCells
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=data,
                                   dtype=tf.float32)
```



### tf.contrib.rnn.LSTMBlockCell
```
__init__(
    num_units,
    forget_bias=1.0,
    cell_clip=None,
    use_peephole=False,
    dtype=None,
    reuse=None,
    name='lstm_cell'
)
```
```
__call__(
    inputs,                       # [batch_size, input_size].
    state,
    scope=None,
    *args,
    **kwargs
)
```
### tf.contrib.rnn.LSTMBlockFusedCell
This is an extremely efficient LSTM implementation, that uses a single TF op for the entire LSTM. It should be both faster and more memory-efficient than LSTMBlockCell defined above.

We add forget_bias (default: 1) to the biases of the forget gate in order to reduce the scale of forgetting in the beginning of the training.
```
__init__(
    num_units,
    forget_bias=1.0,
    cell_clip=None,
    use_peephole=False,
    reuse=None,
    dtype=None,
    name='lstm_fused_cell'
)
```
```
__call__(
    inputs,
    *args,
    **kwargs
)
```
### tf.contrib.rnn.TimeReversedFusedRNN

This is an adaptor to time-reverse a FusedRNNCell
```
cell = tf.contrib.rnn.BasicRNNCell(10)
fw_lstm = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)
bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)
fw_out, fw_state = fw_lstm(inputs)
bw_out, bw_state = bw_lstm(inputs)
```
```
__init__(cell)
```
```
__call__(
    inputs,                # 3-D tensor with shape [time_len x batch_size x input_size]
    initial_state=None,    # [batch_size x state_size]
    dtype=None,
    sequence_length=None,
    scope=None
)
```
### tf.contrib.rnn.stack_bidirectional_dynamic_rnn
Creates a dynamic bidirectional recurrent neural network.

Stacks several bidirectional rnn layers. 
```
The combined forward and backward layer outputs are used as input of the next layer. 
```
tf.bidirectional_rnn does not allow to share forward and backward information between layers. The input_size of the first forward and backward cells must match. The initial state for both directions is zero and no intermediate states are returned.
```
tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
    cells_fw,                    # list of instances of RNNCell
    cells_bw,
    inputs,                      # [Batch_size, max_time, ...]
    initial_states_fw=None,
    initial_states_bw=None,
    dtype=None,
    sequence_length=None,        # contain the actual lengths for each of the sequence.
    parallel_iterations=None,
    time_major=False,            # The shape format of the inputs and outputs Tensors
                                   If true, [max_time, batch_size, depth]
    scope=None
)
```
```
return:
A tuple (outputs, output_state_fw, output_state_bw) 
tensor shaped:
[batch_size, max_time, layers_output]

```
### tf.nn.rnn_cell.MultiRNNCell
```
num_units = [128, 64]
cells = [BasicLSTMCell(num_units=n) for n in num_units]
stacked_rnn_cell = MultiRNNCell(cells)
```
Create a RNN cell composed sequentially of a number of RNNCells
```
__init__(
    cells,              # list of RNNCells that will be composed in this order
    state_is_tuple=True
)
```
Run this RNN cell on inputs, starting from the given state.
```
__call__(
    inputs,                # [batch_size, input_size]
    state,                 # [batch_size, self.state_size]
    scope=None
)
```


## RNN AND EMBEDDING AND SAMPLER 
```
tf.nn.methods:
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

## Static_rnn
```
tf.nn.static_rnn(
    cell,
    inputs,
    initial_state=None,
    dtype=None,
    sequence_length=None,
    scope=None
)
```
```
state = cell.zero_state(...)
  outputs = []
  for input_ in inputs:
    output, state = cell(input_, state)
    outputs.append(output)
  return (outputs, state)
```

## dynamic_rnn
Performs fully dynamic unrolling of inputs
```

tf.nn.dynamic_rnn(
    cell,
    inputs,
    sequence_length=None,
    initial_state=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
```
Examples:
```
# create a BasicRNNCell
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

# defining initial state
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)

```
```
# create 2 LSTMCells
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

# 'outputs' is a tensor of shape [batch_size, max_time, 256]
# 'state' is a N-tuple where N is the number of LSTMCells containing a
# tf.contrib.rnn.LSTMStateTuple for each cell
outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=data,
                                   dtype=tf.float32)
```





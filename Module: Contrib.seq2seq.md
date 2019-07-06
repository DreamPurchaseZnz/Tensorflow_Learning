# Module: tf.contrib.seq2seq

Ops for building neural network seq2seq decoders and losses.

## Classes
```
class AttentionMechanism
class AttentionWrapper         # Wraps another RNNCell with attention.
class AttentionWrapperState    # namedtuple storing the state of a AttentionWrapper.
class BahdanauAttention        # Implements Bahdanau-style (additive) attention.
class BahdanauMonotonicAttention       # Monotonic attention mechanism with Bahadanau-style energy function.
class BasicDecoder                     # Basic sampling decoder.
class BasicDecoderOutput
class BeamSearchDecoder                # BeamSearch sampling decoder.
class BeamSearchDecoderOutput
class BeamSearchDecoderState
class CustomHelper                     # Base abstract class that allows the user to customize sampling.
class Decoder                          # An RNN Decoder abstract interface object.
class FinalBeamSearchDecoderOutput     # Final outputs returned by the beam search after all decoding is finished.
class GreedyEmbeddingHelper            # A helper for use during inference.
class Helper                           # Interface for implementing sampling in seq2seq decoders.
class InferenceHelper                  # A helper to use during inference with a custom sampling function.
class LuongAttention                   # Implements Luong-style (multiplicative) attention scoring.
class LuongMonotonicAttention          # Monotonic attention mechanism with Luong-style energy function.
class SampleEmbeddingHelper            # A helper for use during inference.
class ScheduledEmbeddingTrainingHelper # A training helper that adds scheduled sampling.
class ScheduledOutputTrainingHelper    # A training helper that adds scheduled sampling directly to outputs.
class TrainingHelper                   # A helper for use during training. Only reads inputs.
```

## Functions
```
dynamic_decode                         # Perform dynamic decoding with decoder.
gather_tree                            # Calculates the full beams from the per-step ids and parent beam ids.
hardmax                                # Returns batched one-hot vectors.
monotonic_attention                    # Compute monotonic attention distribution from choosing probabilities.
safe_cumprod(...)                      # Computes cumprod of x in logspace using cumsum to avoid underflow.
sequence_loss(...)                     # Weighted cross-entropy loss for a sequence of logits.
tile_batch(...)                        # Tile the batch dimension of a (possibly nested structure of) tensor(s) t.
```
#### tile_batch
Tile the batch dimension of a (possibly nested structure of) tensor(s) t.
```
tf.contrib.seq2seq.tile_batch(
    t,                             # Tensor shaped [batch_size, ...]
    multiplier, 
    name=None
)
```
For each tensor t in a (possibly nested structure) of tensors, this function takes a tensor t shaped 
```
[batch_size, s0, s1, ...] composed of minibatch entries t[0], ..., t[batch_size - 1] 
```
and tiles it to have a shape 
```
[batch_size * multiplier, s0, s1, ...] 
```
composed of minibatch entries 
```
t[0], t[0], ..., t[1], t[1], ... where each minibatch entry is repeated multiplier times.
```

## AttentionMechanism & AttentionWrapper & AttentionWrapperState

Wraps another RNNCell with attention.

Inherits From: RNNCell
```
__init__(
    cell,                         # RNNCell & MultiRNNCell
    attention_mechanism,          # BahdanauAttention
    attention_layer_size=None,    #  feed the context and cell output into 
                                     the attention layer
                                     to generate attention at each time step
    alignment_history=False,      # store alignment history from all time steps
    cell_input_fn=None,           # concat([inputs, attention], -1)
    output_attention=True,
    initial_cell_state=None,      #  The initial state value to use for the cell 
    name=None,
    attention_layer=None,
    attention_fn=None
)

```
core 
```
def call(self, inputs, state):
    # Step 1
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    # Step 2
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
    # Step 3
    if self._is_multi:
        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history
    else:
        previous_alignments = [state.alignments]
        previous_alignment_history = [state.alignment_history]
    all_alignments = []
    all_attentions = []
    all_histories = []
    for i, attention_mechanism in enumerate(self._attention_mechanisms):
        attention, alignments = _compute_attention(attention_mechanism, cell_output, previous_alignments[i], self._attention_layers[i] if self._attention_layers else None)
        alignment_history = previous_alignment_history[i].write(state.time, alignments) if self._alignment_history else ()
        all_alignments.append(alignments)
        all_histories.append(alignment_history)
        all_attentions.append(attention)
    # Step 4
    attention = array_ops.concat(all_attentions, 1)
    # Step 5
    next_state = AttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(all_histories))
    # Step 6
    if self._output_attention:
        return attention, next_state
    else:
        return cell_output, next_state
```

### Class AttentionWrapperState
namedtuple storing the state of a AttentionWrapper.
```
cell_state                 # The state of the wrapped RNNCell at the previous time step.
attention                  # The attention emitted at the previous time step.
time                       # int32 scalar containing the current time step.
alignments                 # A single or tuple of Tensor(s) containing the alignments emitted at the previous time step 
                             for each attention mechanism.
alignment_history          
attention_state            # A single or tuple of nested objects containing attention mechanism state
                             for each attention mechanism. The objects may contain Tensors or TensorArrays

```
## BahdanauAttention & BahdanauMonotonicAttention & LuongAttention & LuongMonotonicAttention

### BahdanauAttention 
```
__init__(
    num_units,                    # The depth of the query mechanism
    memory,                       # usually the output of an RNN encoder; 
                                    The memory to query;[batch_size, max_time, ...]
    memory_sequence_length=None,  # Sequence lengths for the batch entries in memory
    
    normalize=False,              # Whether to normalize the energy term
    probability_fn=None,          #  Converts the score to probabilities; softmax
    score_mask_value=None, 
    dtype=None,                   # The data type for the query and memory 
                                    layers of the attention mechanism
    custom_key_value_fn=None,
    name='BahdanauAttention'
)
```
#### Methods
```
__call__(
    query,        #  [batch_size, query_depth]
    state         #  [batch_size, alignments_size] (alignments_size is memory's max_time)
)
```
Score the query based on the keys and values, return alignments

This is important for AttentionMechanisms that use the previous alignment to calculate the alignment at the next time step (e.g. monotonic attention)
```
initial_alignments(
    batch_size,
    dtype
)
```
```
Return
A dtype tensor shaped [batch_size, alignments_size] (alignments_size is the values' max_time).
````
### BahdanauMonotonicAttention
This type of attention enforces a monotonic constraint on the attention distributions; that is once the model attends to a given point in the memory it can't attend to any prior points at subsequence output timesteps.

 It achieves this by using the \_monotonic_probability_fn instead of softmax to construct its attention distributions. Since the attention scores are passed through a sigmoid, a learnable scalar bias parameter is applied after the score function and before the sigmoid

```
__init__(
    num_units,                  # The depth of the query mechanism
    memory,
    memory_sequence_length=None,  
    normalize=False,        
    score_mask_value=None,      #  The mask value for score before passing into probability_fn
    sigmoid_noise=0.0,
    sigmoid_noise_seed=None,
    score_bias_init=0.0,
    mode='parallel',            #  How to compute the attention distribution
    dtype=None,
    name='BahdanauMonotonicAttention'
)

```

## BasicDecoder & BasicDecoderOutput
### BasicDecoder
```
__init__(
    cell,
    helper,
    initial_state,
    output_layer=None
)
```
#### Properties

```
batch_size
output_dtype
output_size
```
Describes whether the Decoder keeps track of finished states.
```
tracks_own_finished   
```
#### Methods
```
finalize(
    outputs,
    final_state,
    sequence_lengths
)
```

```
initialize(name=None)
return : 
(finished, first_inputs, initial_state)
```
Perform a decoding step.
```
step(
    time,        # scalar int32 tensor
    inputs, 
    state,
    name=None
)
return:
(outputs, next_state, next_inputs, finished).
```
### BasicdecoderOutput


































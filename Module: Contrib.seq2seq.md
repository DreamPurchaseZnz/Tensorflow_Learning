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







# Bidirectional LSTM with fused LSTM cells.

import tensorflow as tf

class TfModel(object):
    """A deep RNN model, for use of acoustic or duration modeling."""

    def __init__(self, rnn_cell, dnn_depth, dnn_num_hidden, rnn_depth, rnn_num_hidden,
                 output_size, bidirectional=False, rnn_output=False, cnn_output=False,
                 look_ahead=5, mdn_output=False, mix_num=1, name="acoustic_model"):
        self._cell_fn = tf.contrib.rnn.LSTMBlockFusedCell

        self._rnn_cell = rnn_cell
        self._dnn_depth = dnn_depth
        self._dnn_num_hidden = dnn_num_hidden
        self._rnn_depth = rnn_depth
        self._rnn_num_hidden = rnn_num_hidden
        self._output_size = output_size
        self._bidirectional = bidirectional
        self._rnn_output = rnn_output
        self._cnn_output = cnn_output
        self._look_ahead = look_ahead
        self._mdn_output = mdn_output
        self._mix_num = mix_num
        self.collector = {}

        self._input_module = [
            tf.layers.Dense(units=self._dnn_num_hidden,
                            activation=tf.nn.relu,
                            name="linear_input_{}".format(i))
            for i in range(self._dnn_depth)
        ]
        self._rnns = {
            "fw": [
                self._cell_fn(self._rnn_num_hidden,
                              name="{0}_fw_{1}".format(rnn_cell, i))
                for i in range(self._rnn_depth)
            ],
            "bw": [
                tf.contrib.rnn.TimeReversedFusedRNN(
                    self._cell_fn(self._rnn_num_hidden,
                                  name="{0}_bw_{1}".format(rnn_cell, i)))
                for i in range(self._rnn_depth)
            ],
        }
        self._output_module = tf.layers.Dense(output_size, name="linear_output")


    def __call__(self, input_sequence, input_length):

        output_sequence = input_sequence
        output_sequence = tf.transpose(output_sequence, [1, 0, 2])

        fw_new_states, bw_new_states = [], []
        for i in range(self._rnn_depth):
            fw_output, fw_new_state = self._rnns["fw"][i](
                inputs=output_sequence,
                sequence_length=input_length,
                dtype=tf.float32)
            fw_new_states.append(fw_new_state)

            bw_output, bw_new_state = self._rnns["bw"][i](
                inputs=output_sequence,
                sequence_length=input_length,
                dtype=tf.float32)
            bw_new_states.append(bw_new_state)
            output_sequence = tf.concat([fw_output, bw_output], axis=-1)

        final_state_fw = tuple(fw_new_states)
        final_state_bw = tuple(bw_new_states)

        output_sequence = tf.transpose(output_sequence, [1, 0, 2])
        final_state = (final_state_fw, final_state_bw)
        output_sequence_logits = self._output_module(output_sequence)
        return output_sequence_logits, final_state


input_sequence = tf.placeholder(shape=(32, 600, 297), dtype=tf.float32)
input_sequence_length = tf.placeholder(shape=(32,), dtype=tf.int32)
target_sequence =tf.placeholder(shape=(32, 600, 75), dtype=tf.float32)
target_sequence_length = tf.placeholder(shape=(32, ), dtype=tf.int32)


class Object:
    pass


FLAGS = Object()
root = "/mnt/sdb/zhengnianzu/spss-prepoc/generated_king003/"
FLAGS.config_dir = root + 'config/'
FLAGS.data_dir = root + 'data/'
FLAGS.batch_size = 32
FLAGS.input_dim = 297
FLAGS.output_dim = 75
FLAGS.num_threads = 8
FLAGS.rnn_cell = "fused_lstm"
FLAGS.dnn_depth = 3
FLAGS.dnn_num_hidden = 256
FLAGS.rnn_depth = 3
FLAGS.rnn_num_hidden = 256
FLAGS.bidirectional = True
FLAGS.rnn_output = False
FLAGS.cnn_output = False
FLAGS.look_ahead = 5
FLAGS.mdn_output = False
FLAGS.mix_num = 1
FLAGS.learning_rate = 0.001
FLAGS.reduce_learning_rate_multiplier = 0.5
FLAGS.max_grad_norm = 5.0
FLAGS.max_epochs = 30
FLAGS.save_dir = 'exp/acoustic/'
FLAGS.resume_training = False

m = TfModel(
    rnn_cell="fused_lstm",
    dnn_depth=3,
    dnn_num_hidden=256,
    rnn_depth=3,
    rnn_num_hidden=256,
    output_size=75,
    bidirectional=True,
    rnn_output=False,
    cnn_output=False,
    look_ahead=5,
    mdn_output=False,
    mix_num=1,
    name="tf_model")


output_sequence = input_sequence
input_length = input_sequence_length
output_sequence = tf.transpose(output_sequence, [1, 0, 2])





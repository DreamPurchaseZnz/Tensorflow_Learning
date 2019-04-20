class TfModel(object):
    """A deep RNN model, for use of acoustic or duration modeling."""

    def __init__(self, rnn_cell, dnn_depth, dnn_num_hidden, rnn_depth, rnn_num_hidden,
                 output_size, bidirectional=False, rnn_output=False, cnn_output=False,
                 look_ahead=5, mdn_output=False, mix_num=1, name="acoustic_model"):
        """Constructs a TfModel.

        Args:
            rnn_cell: Type of rnn cell including rnn, gru and lstm
            dnn_depth: Number of DNN layers.
            dnn_num_hidden: Number of hidden units in each DNN layer.
            rnn_depth: Number of RNN layers.
            rnn_num_hidden: Number of hidden units in each RNN layer.
            output_size: Size of the output layer on top of the DeepRNN.
            bidirectional: Whether to use bidirectional rnn.
            rnn_output: Whether to use ROL(Rnn Output Layer).
            cnn_output: Whether to use COL(Cnn Output Layer).
            look_ahead: Look ahead window size, used together with cnn_output.
            mdn_output: Whether to interpret last layer as mixture density layer.
            mix_num: Number of gaussian mixes in mdn layer.
            name: Name of the module.
        """

        super(TfModel, self).__init__()

        if rnn_cell == "rnn":
            self._cell_fn = tf.contrib.rnn.BasicRNNCell
        elif rnn_cell == "gru":
            self._cell_fn = tf.contrib.rnn.GRUBlockCellV2
        elif rnn_cell == "lstm":
            self._cell_fn =  tf.contrib.rnn.LSTMBlockCell
        elif rnn_cell == "fused_lstm":
            self._cell_fn = tf.contrib.rnn.LSTMBlockFusedCell
        else:
            raise ValueError("model type not supported: {}".format(rnn_cell))

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

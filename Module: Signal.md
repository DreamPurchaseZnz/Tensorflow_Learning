# Signal

## Frame
Expands signal's axis dimension into frames of frame_length.

Slides a window of size frame_length over signal's axis dimension with a stride of frame_step, 
replacing the axis dimension with [frames, frame_length] frames.

If pad_end is True, window positions that are past the end of the axis dimension are padded with pad_value 
until the window moves fully past the end of the dimension. Otherwise, only window positions 
that fully overlap the axis dimension are produced.

```
tf.signal.frame(
    signal,
    frame_length,
    frame_step,
    pad_end=False,
    pad_value=0,
    axis=-1,
    name=None
)
```

```
pcm = tf.placeholder(tf.float32, [None, 9152])
frames = tf.signal.frame(pcm, 512, 180)
magspec = tf.abs(tf.signal.rfft(frames, [512]))
image = tf.expand_dims(magspec, 3)
```

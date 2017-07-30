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
Methods:
```
add_event
add_graph
add_meta_graph           --->  Adds a MetaGraphDef to the event file.
add_run_metadata
add_session_log
add_summary              --->  Adds a Summary protocol buffer to the event file
close                    --->  Call this method when you do not need the summary writer anymore
flush
get_logdir
reopen                   --->  can be called after close() to add more events in the same directory. The events will go into a new                                events file
```
Now that you've modified your graph and have a FileWriter, you're ready to start running your network! If you want, you could run the merged summary op every single step, and record a ton of training data. That's likely to be more data than you need, though. Instead, consider running the merged summary op every n steps.

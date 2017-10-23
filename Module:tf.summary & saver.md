Module: tf.summary
---------------------------------------------------------------------------------------------------
## Summary Operations
Tensor summaries for exporting information about a model. 
tf.summary
Method:
```
# data type
scalar                   --->  A single scalar value.
tensor_summary 
audio                    
image                    --->  Adding a image summary
histogram                --->  visualize your data's distribution 
text                     --->  textual data

# operation 
merge                    --->  Merges summaries
merge_all                --->  Merges all summaries collected 
filewriter

summary_description
get_summary_description  
```
e.g.
```

tf.reset_default_graph()
# None provide placeholder for batch size
v = tf.placeholder(dtype = tf.float32,shape =[None,1])
# summary.scalar must be a numerical value
a = tf.add(v,1)
a_s=tf.summary.scalar('s1',tf.reduce_mean(a))
b = tf.add(v,2)
b_s=tf.summary.scalar('s2',tf.reduce_mean(b))
c = tf.add(v,3)
c_s=tf.summary.scalar('s3',tf.reduce_mean(c))

merge_all = tf.summary.merge_all()
merge_ab  = tf.summary.merge(a_s,b_s)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(a,feed_dict={v:[[1],[2]]})
sess.run(merge_all,feed_dict={v:[[1]]})
```

## Visualize the status
In order to emit the events files used by Tensorboard, all the summaries were **collected** into a single Tensor during **graph building phase**
```
summary = tf.summary.merge_all()
```
After the Session is created, a **tf.summary.filewriter** may be **instantiated** to **write** event files,which contain both graph iteself and the values of the summaries
``` 
summary_writer = tf.summary.FileWriter(<log_dir>,<sess.graph>)
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
Lastly, the event file will be updatad with new summary value **every time the summary** is evaluated and the output passed to the writer's add_summary method 
```
summary_str = sess.run(summary, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)
```
When the event files are written, Tensorboard may be run against the training folder to display the values from summaries.
There is another way to add summary
```
    def log_loss_accuracy(self, loss, accuracy, epoch, prefix, should_print=True):
        if should_print:
            print('mean cross_entropy: %f,mean accuracy:%f' % (
                loss, accuracy
            ))

        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)
            ),
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(accuracy)
            )
        ])
        self.summary_writer.add_summary(summary, epoch)
```
Compared with summary_str, this way is much like the normal way, First it does not need define the summary string and the operation of
running the graph again, Second it's much easier, however it needs some extra knowledge, because this point is not covered by the 
cookbook





Module: tf.train.saver()
---------------------------------------------------------------------------------------------------
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

restore                                --->  A way to initialize variables
save                                   --->  Save variables,it requires a session in which graph was launched and variables was                                                initialized


set_last_checkpoints_with_time         --->  Sets the list of old checkpoint filenames and timestamps
to_proto

```
## Save a checkpoint
In order to emit a checkpoint file that may be used to **later restore a model** for **further training or evaluation**, we can instantiate a tf.train.saver.
```
saver = tf.train.Saver()
```
In the training loop, the *tf.train.Saver.save* method will **periodically** be called to write a checkpoint file to the training dictionary with the current values of **all the trianable variable**.
```
saver.save(sess, ckpt_dir,global_step= step)
```
At some later point in the future, training might be **resummed** by using the *tf.train.saver.restore* method to **reload the model parameters**.
```
saver.restore(sess,tf.train.latest_checkpoint(ckpt_dir))
```
Restore previously saved variables ,It requires **a session** in which **the graph** was launched. 
The variables to restore do not have to have been initialized, as restoring is itself **a way to initialize variables**.

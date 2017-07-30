Module: Visualize the status
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

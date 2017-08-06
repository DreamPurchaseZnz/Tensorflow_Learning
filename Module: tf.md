Module: tf
-----------------------------------------------------------------------------------
```
tf.reshape
tf.placeholder
tf.Variable

tf.tuple
tf.group
tf.random_uniform
tf.reduce_mean
tf.global_norm
tf.IndexedSlices
tf.assign
tf.clip_by_value
tf.transpose
tf.expand_dims                               ---> Insert a dimension of 1  into a tensor'shape
tf.squeeze                                   ---> Removes dimensions of size 1 from the shape of a tensor
tf.tile                                      ---> Construct a tenosr by tiling a given tensor
tf.concat                                    ---> Concatenates tensors along one dimension
tf.minimum                                   ---> Returns the min of x and y (i.e. x < y ? x : y) element-wise.
tf.maximum                                     
```
e.g.
f:(batch_size,y_dim)  ---> (batch_size,img_dim,img_dim ,y_dim)
```
# Smart method
f = tf.expand_dims(tf.expand_dims(f, 1), 2)
f = tf.tile(f, [1, img_dim , img_dim , 1])

# Stupid method
f=[]
for i in range(y_dim):
  g = tf.tile(f(:,i),[batch_size,img_dim]).reshape(batch_size,1,img_dim,1)
  f.append(g)
f= tf.concat(g,axis = 0)

```

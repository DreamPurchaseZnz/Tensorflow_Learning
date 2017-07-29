# Sharing variables
--------------------------------------------------------------------------------------------
Today, we will talk something about sharing variables . It can be a very complex problem,however ,it is actually neccessary to know that,for when building complex models , you ofen need to share large of variables and you might want to initialize all of them in one place
so this tutorial show how this can be done using the following functions 
```
tf.variable_scope                   ---> Carry a name that will be used as a prefix for variables names and a reuse flag to         
                                         Distinguish the function of get_variable
tf.get_variable                     ---> Create new variables or reusing variables
tf.reset_default_graph()            ---> Clears the default graph stack and resets the global default graph
tf.get_variable_scope()             ---> Retrieve the current scope
tf.get_variable_scope.reuse_variables()  ---> Set reuse flag to be true
```
## Necessity
```
import tensorflow as tf
def my_image_filter(input_images):
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),
        name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + conv2_biases)
    
img1 = tf.placeholder(tf.float32,shape=(100,32,32,32))
result1 = my_image_filter(img1)
result2 = my_image_filter(img1)
```
**Problem** is that you call my_image_filter twice,but this will create two sets of variables,4 variable in each one, for a total of 8 variables,there is a solution, tedious but efficent ,by using variable_dict.
```
variables_dict = {
    "conv1_weights": tf.Variable(tf.random_normal([5, 5, 32, 32]),name="conv1_weights"),
    "conv1_biases": tf.Variable(tf.zeros([32]), name="conv1_biases"),
    "conv2_weights": tf.Variable(tf.random_normal([5, 5, 32, 32]),name="conv2_weights"),
    "conv2_biases": tf.Variable(tf.zeros([32]), name="conv2_biases")}
    
def my_image_filter(input_images, variables_dict):
    conv1 = tf.nn.conv2d(input_images, variables_dict["conv1_weights"],
        strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + variables_dict["conv1_biases"])

    conv2 = tf.nn.conv2d(relu1, variables_dict["conv2_weights"],
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + variables_dict['conv2_biases'])
    
 # Both calls to my_image_filter() now use the same variables
result1 = my_image_filter(img1, variables_dict)
result2 = my_image_filter(img1, variables_dict)
```
With convenient ,creating varibles like above ,outside of the code,breaks encapsulation
* the code builds graph must document the names ,shapes of variable to create.
* when the code changes ,the callers may have to create more , less or different varibles

So More general solution have been proposed by using Variable Scope mechanism that allows to easily share named variables while constructing a graph.
```
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
    
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
        
result1 = my_image_filter(img1)
result2 = my_image_filter(img1) # it will raise a error that (Variable conv1/weights already exists, disallowed...)

# this will be ok
with tf.variable_scope("image_filters") as scope:
    result1 = my_image_filter(img1)
    scope.reuse_variables()
    result2 = my_image_filter(img1)
```
## Variable_scope
Variable_scope determinate the function of tf.get_variable
* case 1 : The scope is set for creating new variables ,as evidenced by tf.get_variable_scope.reuse() = false
```
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
```
* case 2 : The scope is set for reusing variables as evidenced by tf.get_variable_scope.reuse()=True
```
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 is v
```
The primary function of variable scope is to carry a name that will be used as **prefix** for variables names and a **reuse-flag** to distinguish the two cases discribed above. 
nesting variable scopes append their names in a way analogous to how directories work

the current scope can be **retieved** by using tf.get_variable_scope 
and a reuse flag can be set to **True** by tf.get_variable_scope.reuse_variables()
```
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
assert v.name == "foo/bar/v:0"

with tf.variable_scope("foo"):
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 is v
```
Obviously ,you cannot set the reuse flag to false. but you can **enter a reusing variable scope** and then exit it, then exist it ,going back to a non_reusing one 
* using a paramster reuse = True when opening a variable scope
* the reusing parameter is inherited
```
with tf.variable_scope("root"):
    # At start, the scope is not reusing.
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo"):
        # Opened a sub-scope, still not reusing.
        assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo", reuse=True):
        # Explicitly opened a reusing scope.
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope("bar"):
            # Now sub-scope inherits the reuse flag.
            assert tf.get_variable_scope().reuse == True
    # Exited the reusing scope, back to a non-reusing one.
    assert tf.get_variable_scope().reuse == False
```
**Error** arise when:
* case 1: when variable_scope is set reuse=True  ,cannot get exist variable.
* case 2: when variable_scope is set reuse=False ,the new variable have been exist.

## Name_scope vs Variable_scope
when we do with tf.variable_scope('name') ,this implictly opens a tf.name_scope
* variable_scope to govern names of variable.
* name_scope only affect the the names of ops, but not of variables. 
```
tf.reset_default_graph()
with tf.variable_scope("foo"):
    x = 1.0 + tf.get_variable("v", [1])
assert x.op.name == "foo/add"
```

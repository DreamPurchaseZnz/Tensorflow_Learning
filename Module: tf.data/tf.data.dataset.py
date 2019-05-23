
import tensorflow as tf
import numpy as np


x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))



# using two numpy arrays
features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))

# using a tensor
dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(el))


# using a placeholder
x = tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)

data = np.random.sample((100,2))

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer, feed_dict={ x: data })
    print(sess.run(el))


# from generator
sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])

def generator():
    for el in sequence:
        yield el

dataset = tf.data.Dataset().batch(1).from_generator(generator,
                                           output_types= tf.int64,
                                           output_shapes=(tf.TensorShape([None, 1])))

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))


# initializable iterator to switch between data
EPOCHS = 10

x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
dataset = tf.data.Dataset.from_tensor_slices((x, y))

train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()

with tf.Session() as sess:
#     initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1]})
    for _ in range(EPOCHS):
        sess.run([features, labels])
#     switch to test data
    sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1]})
    print(sess.run([features, labels]))


# Reinitializable iterator to switch between Datasets
EPOCHS = 10
# making fake data using numpy
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))
# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
features, labels = iter.get_next()
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)

with tf.Session() as sess:
    sess.run(train_init_op) # switch to train dataset
    for _ in range(EPOCHS):
        sess.run([features, labels])
    sess.run(test_init_op) # switch to val dataset
    print(sess.run([features, labels]))

# feedable iterator to switch between iterators
EPOCHS = 10
# making fake data using numpy
train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
test_data = (np.random.sample((10, 2)), np.random.sample((10, 1)))
# create placeholder
x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])
# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
test_dataset = tf.data.Dataset.from_tensor_slices((x, y))
# create the iterators from the dataset
train_iterator = train_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()
# same as in the doc https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
handle = tf.placeholder(tf.string, shape=[])
iter = tf.data.Iterator.from_string_handle(
    handle, train_dataset.output_types, train_dataset.output_shapes)
next_elements = iter.get_next()

with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    # initialise iterators. In our case we could have used the 'one-shot' iterator instead,
    # and directly feed the data insted the Dataset.from_tensor_slices function, but this
    # approach is more general
    sess.run(train_iterator.initializer, feed_dict={x: train_data[0], y: train_data[1]})
    sess.run(test_iterator.initializer, feed_dict={x: test_data[0], y: test_data[1]})

    for _ in range(EPOCHS):
        x, y = sess.run(next_elements, feed_dict={handle: train_handle})
        print(x, y)

    x, y = sess.run(next_elements, feed_dict={handle: test_handle})
    print(x, y)


# BATCHING
BATCH_SIZE = 4
x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))


# REPEAT
BATCH_SIZE = 4
x = np.array([[1],[2],[3],[4]])
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.repeat()

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    for _ in range(8):
        print(sess.run(el))


# MAP
x = np.array([[1],[2],[3],[4]])
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.map(lambda x: x*2)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
#     this will run forever
        for _ in range(len(x)):
            print(sess.run(el))


# SHUFFLE
BATCH_SIZE = 4
x = np.array([[1],[2],[3],[4]])
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    print(sess.run(el))



# how to pass the value to a model
EPOCHS = 10
BATCH_SIZE = 16
# using two numpy arrays
features, labels = (np.array([np.random.sample((100,2))]),
                    np.array([np.random.sample((100,1))]))

dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)

iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()

# make a simple model
net = tf.layers.dense(x, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, y) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))


# Wrapping all together -> Switch between train and test set using Initializable iterator
EPOCHS = 10
# create a placeholder to dynamically switch between batch sizes
batch_size = tf.placeholder(tf.int64)
BATCH_SIZE = 32

x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()

# using two numpy arrays
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((20,2)), np.random.sample((20,1)))

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
# make a simple model
net = tf.layers.dense(features, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, labels) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

n_batches = train_data[0].shape[0] // BATCH_SIZE

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
    print('Training...')
    for i in range(EPOCHS):
        tot_loss = 0
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
    # initialise iterator with test data
    sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
    print('Test Loss: {:4f}'.format(sess.run(loss)))


# Wrapping all together -> Switch between train and test set using Reinitializable iterator
EPOCHS = 10
# create a placeholder to dynamically switch between batch sizes
batch_size = tf.placeholder(tf.int64)

x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(batch_size) # always batch even if you want to one shot it
# using two numpy arrays
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((20,2)), np.random.sample((20,1)))

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
features, labels = iter.get_next()
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)

# make a simple model
net = tf.layers.dense(features, 8, activation=tf.tanh) # pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)

loss = tf.losses.mean_squared_error(prediction, labels) # pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(train_init_op, feed_dict = {x : train_data[0], y: train_data[1], batch_size: 16})
    print('Training...')
    for i in range(EPOCHS):
        tot_loss = 0
        for _ in range(n_batches):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / n_batches))
    # initialise iterator with test data
    sess.run(test_init_op, feed_dict = {x : test_data[0], y: test_data[1], batch_size:len(test_data[0])})
    print('Test Loss: {:4f}'.format(sess.run(loss)))


# load a csv
CSV_PATH = './tweets.csv'
dataset = tf.contrib.data.make_csv_dataset(CSV_PATH, batch_size=32)
iter = dataset.make_one_shot_iterator()
next = iter.get_next()
print(next) # next is a dict with key=columns names and value=column data
inputs, labels = next['text'], next['sentiment']

with  tf.Session() as sess:
    print(sess.run([inputs,labels]))

log_time = {}


# copied form https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def how_much(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__)
            kw['log_time'][name] = (te - ts)

        return result

    return timed


# benchmark
import time
DATA_SIZE = 5000
DATA_SHAPE = ((32,32),(20,))
BATCH_SIZE = 64
N_BATCHES = DATA_SIZE // BATCH_SIZE
EPOCHS = 10

test_size = (DATA_SIZE//100)*20

train_shape = ((DATA_SIZE, *DATA_SHAPE[0]),(DATA_SIZE, *DATA_SHAPE[1]))
test_shape = ((test_size, *DATA_SHAPE[0]),(test_size, *DATA_SHAPE[1]))
print(train_shape, test_shape)
train_data = (np.random.sample(train_shape[0]), np.random.sample(train_shape[1]))
test_data = (np.random.sample(test_shape[0]), np.random.sample(test_shape[1]))

# used to keep track of the methodds
log_time = {}

tf.reset_default_graph()
sess = tf.InteractiveSession()

input_shape = [None, *DATA_SHAPE[0]]  # [None, 64, 64, 3]
output_shape = [None, *DATA_SHAPE[1]]  # [None, 20]
print(input_shape, output_shape)

x, y = tf.placeholder(tf.float32, shape=input_shape), tf.placeholder(tf.float32, shape=output_shape)


@how_much
def one_shot(**kwargs):
    print('one_shot')
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(BATCH_SIZE).repeat()
    train_el = train_dataset.make_one_shot_iterator().get_next()

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(BATCH_SIZE).repeat()
    test_el = test_dataset.make_one_shot_iterator().get_next()
    for i in range(EPOCHS):
        print(i)
        for _ in range(N_BATCHES):
            sess.run(train_el)
        for _ in range(N_BATCHES):
            sess.run(test_el)


@how_much
def initialisable(**kwargs):
    print('initialisable')
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE).repeat()

    iter = dataset.make_initializable_iterator()
    elements = iter.get_next()

    for i in range(EPOCHS):
        print(i)
        sess.run(iter.initializer, feed_dict={x: train_data[0], y: train_data[1]})
        for _ in range(N_BATCHES):
            sess.run(elements)
        sess.run(iter.initializer, feed_dict={x: test_data[0], y: test_data[1]})
        for _ in range(N_BATCHES):
            sess.run(elements)


@how_much
def reinitializable(**kwargs):
    print('reinitializable')
    # create two datasets, one for training and one for test
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE).repeat()
    # create a iterator of the correct shape and type
    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
    elements = iter.get_next()
    # create the initialisation operations
    train_init_op = iter.make_initializer(train_dataset)
    test_init_op = iter.make_initializer(test_dataset)

    for i in range(EPOCHS):
        print(i)
        sess.run(train_init_op, feed_dict={x: train_data[0], y: train_data[1]})
        for _ in range(N_BATCHES):
            sess.run(elements)
        sess.run(test_init_op, feed_dict={x: test_data[0], y: test_data[1]})
        for _ in range(N_BATCHES):
            sess.run(elements)


@how_much
def feedable(**kwargs):
    print('feedable')
    # create two datasets, one for training and one for test
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE).repeat()
    test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE).repeat()
    # create the iterators from the dataset
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iter = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    elements = iter.get_next()

    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    sess.run(train_iterator.initializer, feed_dict={x: train_data[0], y: train_data[1]})
    sess.run(test_iterator.initializer, feed_dict={x: test_data[0], y: test_data[1]})

    for i in range(EPOCHS):
        print(i)
        for _ in range(N_BATCHES):
            sess.run(elements, feed_dict={handle: train_handle})
        for _ in range(N_BATCHES):
            sess.run(elements, feed_dict={handle: test_handle})


one_shot(log_time=log_time)
initialisable(log_time=log_time)
reinitializable(log_time=log_time)
feedable(log_time=log_time)

sorted((value, key) for (key, value) in log_time.items())


















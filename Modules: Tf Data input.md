# TensorFlow Data Input : Placeholders, Protobufs & Queues
The processes of getting data into a model can be rather annoying, with a lot of glue code. 

TensorFlow tries to fix this by providing a few ways to feed in data. 
The easiest of these is to use 
```
placeholders
```
which allow you to manually pass in numpy arrays of data.

The second method, my preferred, is to do as much as I possibly can on the graph and to make use of 
```
binary files and input queues
```
Not only does this lighten the amount of code I need to write, removing the need to do any data augmentation or file reading, but the interface is reasonably standard across different kinds of data. It is also conceptually cleaner. 

## Placeholders

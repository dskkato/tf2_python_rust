import tensorflow
import tensorflow.compat.v1 as tf

graph = tensorflow.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float32, name="x")
    y = tf.placeholder(tf.float32, name="y")

    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="w")
    b = tf.Variable(tf.zeros([1]), name="b")
    y_hat = w * x + b

    loss = tf.reduce_mean(tf.square(y_hat - y))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss, name="train")

    init = tf.variables_initializer(tf.global_variables(), name="init")

    # Declare saver ops
    saver = tf.train.Saver(tf.global_variables())

directory = "examples/regression_checkpoint"
tf.train.write_graph(graph, directory, "model.pb", as_text=False)

logdir = "logs/regression_checkpoint"
writer = tensorflow.summary.create_file_writer(logdir)
with writer.as_default():
    tensorflow.summary.graph(graph)

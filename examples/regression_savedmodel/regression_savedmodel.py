import tensorflow as tf


class LinearRegresstion(tf.Module):
    def __init__(self):
        w = tf.Variable(tf.random.uniform([1], -1.0, 1.0), name="w")
        b = tf.Variable(tf.zeros([1]), name="b")
        self.w = w
        self.b = b
        self.trainable_weights = [w, b]
        self.optimizer = tf.optimizers.SGD(0.5)

    @tf.function
    def __call__(self, x):
        y_hat = self.w * x + self.b
        return y_hat, self.w, self.b

    @tf.function
    def get_weights(self):
        return self.trainable_weights

    @tf.function
    def train(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self(x)[0]
            loss = tf.reduce_mean(tf.square(y_hat - y))
        grads = tape.gradient(loss, self.trainable_weights)
        train = self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights), name="train"
        )
        return loss


model = LinearRegresstion()

x = tf.TensorSpec([None], tf.float32, name="x")
y = tf.TensorSpec([None], tf.float32, name="y")
train = model.train.get_concrete_function(x, y)
weights = model.get_weights.get_concrete_function()

directory = "examples/regression_savedmodel"
signatures = {"train": train, "weights": weights}
tf.saved_model.save(model, directory, signatures=signatures)

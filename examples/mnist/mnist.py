# train mnist model
# ref https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
import numpy as np
from numpy.testing import assert_array_equal

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
sample = x_test[0]  # keep first test item to verify Rust version later on
x_train, x_test = x_train / 255.0, x_test / 255.0

# explicitly constrain input type as float32
x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

# dump the test item
buf = tf.image.encode_png(sample[:, :, tf.newaxis])
tf.io.write_file("examples/mnist/sample.png", buf)

# construct a training model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1)

# convert output type through softmax so that it can be interpreted as probability
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# convert keras model to TF2 function to get a computation graph
x = tf.TensorSpec((None, 28, 28), tf.float32)
tf_model = tf.function(lambda x: probability_model(x)).get_concrete_function(x=x)

# now all variables are converted to constants.
# if this step is omitted, dumped graph does not include trained weights
frozen_model = convert_variables_to_constants_v2(tf_model)

# verify the outputs are unchanged through above conversions
expected = probability_model(x_test[:1])
tf_model_output = tf_model(x_test[:1])
frozen_model_output = frozen_model(tf.convert_to_tensor(x_test[:1]))
assert_array_equal(expected, tf_model_output)
assert_array_equal(expected, frozen_model_output[0])

# write the computation graph and weights
directory = "examples/mnist"
tf.io.write_graph(frozen_model.graph, directory, "model.pb", as_text=False)

# dump expected values to compare Rust's outputs
with open("examples/mnist/expected_values.txt", "w") as f:
    values = expected.numpy()[0]
    print(*values, sep=", ", file=f)

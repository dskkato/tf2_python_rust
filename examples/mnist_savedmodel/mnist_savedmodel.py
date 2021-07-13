# train mnist model
# ref https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
sample = x_test[0]  # keep first test item to verify Rust version later on
x_train, x_test = x_train / 255.0, x_test / 255.0

# explicitly constrain input type as float32
x_train, x_test = x_train.astype("float32"), x_test.astype("float32")

# dump the test item
buf = tf.image.encode_png(sample[:, :, tf.newaxis])
tf.io.write_file("examples/mnist_savedmodel/sample.png", buf)

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
inputs = tf.keras.Input((28, 28), name="input", dtype=tf.float32)
x = model(inputs)
outputs = tf.keras.layers.Softmax(name="output")(x)

probability_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# dump expected values to compare Rust's outputs
with open("examples/mnist_savedmodel/expected_values.txt", "w") as f:
    values = probability_model(x_test[:1, :, :])[0].numpy()
    print(*values, sep=", ", file=f)

directory = "examples/mnist_savedmodel"
tf.saved_model.save(probability_model, directory)

# export graph info to TensorBoard
logdir = "logs/mnist_savedmodel"
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on()
values = probability_model(x_test[:1, :, :])
with writer.as_default():
    tf.summary.trace_export("Default", step=0)

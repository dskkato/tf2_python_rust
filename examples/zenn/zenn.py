import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)
from numpy.testing import assert_almost_equal

# default input shape 224x224x3
model = tf.keras.applications.MobileNetV3Large()

x = tf.TensorSpec(model.input_shape, tf.float32, name="x")
concrete_function = tf.function(lambda x: model(x)).get_concrete_function(x)
# now all variables are converted to constants.
# if this step is omitted, dumped graph does not include trained weights
frozen_model = convert_variables_to_constants_v2(concrete_function)
directory = "examples/zenn"
tf.io.write_graph(frozen_model.graph, directory, "mobilenetv3large.pb", as_text=False)

# original prediction
buf = tf.io.read_file("examples/zenn/sample.png")
img = tf.image.decode_png(buf)
sample = img[tf.newaxis, :, :, :]
original_pred = model(sample)

frozen_model_pred = frozen_model(tf.cast(sample, tf.float32))[0]
assert_almost_equal(original_pred.numpy(), frozen_model_pred.numpy())
decoded = tf.keras.applications.mobilenet_v3.decode_predictions(original_pred.numpy())[
    0
]
print(decoded)

import tensorflow as tf

# default input shape 224x224x3
model = tf.keras.applications.MobileNetV3Large()

directory = "examples/zenn_savedmodel"
model.save(directory)

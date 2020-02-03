import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"

# disable logging
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import multiprocessing
import tensorflow.compat.v2 as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import tensorflow_datasets as tfds

import model_v2

USE_AMP = True
USE_XLA = True

BATCH_SIZE = 180
IMG_SIZE = (224, 224)
IMG_DTYPE = tf.float16
IMG_DTYPE_ = "float16"

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

# Load Dataset

splits = tfds.Split.TRAIN.subsplit(weighted=(1, 0, 0))

(raw_train, raw_validation, raw_test), info = tfds.load("cats_vs_dogs",
                                                        split=list(splits),
                                                        with_info=True,
                                                        as_supervised=True)

num_class = 2
num_examples = info.splits["train"].num_examples
num_train = int(num_examples * 1.0)

print("")
print("Number of training examples:", num_train)
print("")

@tf.function
def format_example(image, label):
    """
    This function will run as part of a tf.data pipeline.
    It is reponsible for resizing and normalizing the input images.
    """
    image = tf.image.resize(image, IMG_SIZE)
    image = (image/127.5) - 1
    image = tf.cast(image, IMG_DTYPE)
    label = tf.one_hot(label, num_class)
    return image, label

train = raw_train.shuffle(4096)
train = train.repeat(count=-1)
train = train.map(format_example, num_parallel_calls=multiprocessing.cpu_count())
train = train.batch(BATCH_SIZE)
train = train.prefetch(10)

# Build Model
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE *= strategy.num_replicas_in_sync

if USE_AMP:
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    
with strategy.scope():
    model = model_v2.create_model(IMG_SIZE, IMG_DTYPE_, num_class)
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"])

# Train Model

time_callback = model_v2.TimeHistory()

steps_per_epoch = 100

model.fit(train, steps_per_epoch=steps_per_epoch,
          epochs=3, callbacks=[time_callback], verbose=2)

duration = min(time_callback.times)
fps = steps_per_epoch*BATCH_SIZE/duration
fps_per_gpu = fps/strategy.num_replicas_in_sync

print("\nResults:\n")
print("ResNet 50 FPS:")
print(strategy.num_replicas_in_sync, "GPU:", int(fps))
print("Per GPU:", int(fps_per_gpu))

SAVE_DIR = "./model.h5"
model.save(SAVE_DIR)
print("Model saved to:", SAVE_DIR)

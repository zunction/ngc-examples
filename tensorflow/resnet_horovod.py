import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rn152", action="store_true", default=False,
                    help="Train a larger ResNet-152 model instead of ResNet-50")
parser.add_argument("--amp", action="store_true", default=False,
                    help="Use grappler AMP for mixed precision training")
parser.add_argument("--keras_amp", action="store_true", default=False,
                    help="Use Keras AMP for mixed precision training")
parser.add_argument("--xla", action="store_true", default=False,
                    help="Use XLA compiler")
parser.add_argument("--fp16comp", action="store_true", default=True,
                    help="Use float16 compression during allreduce")
args = parser.parse_args()

import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"

import multiprocessing
import tensorflow.compat.v2 as tf
import horovod.tensorflow.keras as hvd

hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[hvd_rank], "GPU")

import tensorflow_datasets as tfds
import tf_models
import utils

LEARNING_RATE = 0.001
BATCH_SIZE = 80
IMG_SIZE = (224, 224)

print("Using XLA:", args.xla)
tf.config.optimizer.set_jit(args.xla)
print("Using grappler AMP:", args.amp)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp})

print("Loading Dataset")

dataset, info = tfds.load("cats_vs_dogs",
                          with_info=True,
                          as_supervised=True)

num_class = 2
num_examples = info.splits["train"].num_examples
num_train = int(num_examples * 1.0)

print("Number of training examples:", num_train)

@tf.function
def format_example(image, label):
    """
    This function will run as part of a tf.data pipeline.
    It is reponsible for resizing and normalizing the input images.
    """
    image = tf.image.resize(image, IMG_SIZE)
    image = (image/127.5) - 1
    label = tf.one_hot(label, num_class)
    return image, label

print("Build tf.data input pipeline")

train = dataset["train"].shard(num_shards=hvd_size, index=hvd_rank)
train = train.shuffle(100000)
train = train.repeat(count=-1)
train = train.map(format_example, num_parallel_calls=int(multiprocessing.cpu_count()/hvd_size)-1)
train = train.batch(BATCH_SIZE)
train = train.prefetch(64)

print("Output:", str(train.take(1)))

print("Build and distribute model")

LEARNING_RATE *= hvd_size**0.5

if hvd_rank == 0:
    print("Number of devices:", hvd_size)
    print("Global batch size:", BATCH_SIZE*hvd_size)
    print("Adjusted learning rate:", LEARNING_RATE)

if args.keras_amp:
    print("Using Keras AMP:", args.keras_amp)
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    
if args.rn152:
    print("Using ResNet-152 model")
    model = tf_models.rn152(IMG_SIZE, num_class)
else:
    print("Using ResNet-50 model")
    model = tf_models.rn50(IMG_SIZE, num_class)
opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
if args.amp:
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
if args.fp16comp:
    print("Using float16 compression for all-reduce")
    compression = hvd.Compression.fp16
else:
    compression = hvd.Compression.none
opt = hvd.DistributedOptimizer(opt, compression=compression)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["acc"],
              experimental_run_tf_function=False)

print("Train model")

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
]

if hvd_rank == 0:
    verbose = 2
    time_callback = utils.TimeHistory()
    callbacks.append(time_callback)
else:
    verbose = 0

steps_per_epoch = int(num_train/BATCH_SIZE/hvd_size)
model.fit(train, steps_per_epoch=steps_per_epoch,
          epochs=5, callbacks=callbacks, verbose=verbose)

if hvd_rank == 0:
    duration = min(time_callback.times)
    fps = steps_per_epoch*BATCH_SIZE/duration
    
    print("\nResults:\n")
    print("ResNet FPS:")
    print("*", hvd_size, "GPU:", int(fps*hvd_size))
    print("* Per GPU:", int(fps))

    SAVE_DIR = "./model.h5"
    model.save(SAVE_DIR)
    print("Model saved to:", SAVE_DIR)

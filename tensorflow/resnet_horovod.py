import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rn152", action="store_true", default=False,
                    help="Train a larger ResNet-152 model instead of ResNet-50")
parser.add_argument("--dn201", action="store_true", default=False,
                    help="Train a larger DenseNet-201 model instead of ResNet-50")
parser.add_argument("--mobilenet", action="store_true", default=False,
                    help="Train a smaller MobileNetV2 model instead of ResNet-50")
parser.add_argument("--amp", action="store_true", default=False,
                    help="Use grappler AMP for mixed precision training")
parser.add_argument("--keras_amp", action="store_true", default=False,
                    help="Use Keras AMP for mixed precision training")
parser.add_argument("--xla", action="store_true", default=False,
                    help="Use XLA compiler")
parser.add_argument("--fp16comp", action="store_true", default=False,
                    help="Use float16 compression during allreduce")
parser.add_argument("--batchsize", default=128, type=int,
                    help="Batch size to use for training")
parser.add_argument("--imgsize", default=224, type=int,
                    help="Image size to use for training")
parser.add_argument("--lr", default=0.001, type=float,
                    help="Learning rate")
parser.add_argument("--epochs", default=40, type=int,
                    help="Number of epochs to train for")
args = parser.parse_args()

import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARN)
import time
import multiprocessing
import tensorflow.compat.v2 as tf
import horovod.tensorflow.keras as hvd

hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[hvd_rank], "GPU")
tf.config.experimental.set_memory_growth(gpus[hvd_rank], True)

import tensorflow_datasets as tfds
import cnn_models
import utils

LEARNING_RATE = args.lr
BATCH_SIZE = args.batchsize
IMG_SIZE = (args.imgsize, args.imgsize)
EPOCHS = args.epochs

print("Using XLA:", args.xla)
tf.config.optimizer.set_jit(args.xla)
print("Using grappler AMP:", args.amp)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp})

print("Loading Dataset")

dataset, info = tfds.load("imagenette/320px",
                          with_info=True,
                          as_supervised=True)

num_class = 10
num_train = info.splits["train"].num_examples
num_valid = info.splits["validation"].num_examples

print("Number of training examples:", num_train)
print("Number of validation examples:", num_valid)

@tf.function
def format_train_example(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = (image/127.5) - 1
    label = tf.one_hot(label, num_class)
    return image, label

@tf.function
def format_test_example(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = (image/127.5) - 1
    label = tf.one_hot(label, num_class)
    return image, label

print("Build tf.data input pipeline")

train = dataset["train"].shard(num_shards=hvd_size, index=hvd_rank)
train = train.shuffle(100000)
train = train.repeat(count=-1)
train = train.map(format_train_example, num_parallel_calls=int(multiprocessing.cpu_count()/hvd_size)-1)
train = train.batch(BATCH_SIZE, drop_remainder=True)
train = train.prefetch(64)

valid = dataset["validation"].shard(num_shards=hvd_size, index=hvd_rank)
valid = valid.repeat(count=-1)
valid = valid.map(format_test_example, num_parallel_calls=int(multiprocessing.cpu_count()/hvd_size)-1)
valid = valid.batch(BATCH_SIZE//2, drop_remainder=False)
valid = valid.prefetch(64)

print("Output:", str(train.take(1)), str(valid.take(1)))

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
    model = cnn_models.rn152(IMG_SIZE, num_class, weights=None)
elif args.dn201:
    print("Using DenseNet-201 model")
    model = cnn_models.dn201(IMG_SIZE, num_class, weights=None)
elif args.mobilenet:
    print("Using MobileNetV2 model")
    model = cnn_models.mobilenet(IMG_SIZE, num_class, weights=None)
else:
    print("Using ResNet-50 model")
    model = cnn_models.rn50(IMG_SIZE, num_class, weights=None)
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
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=5, end_epoch=20, multiplier=1.),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=20, end_epoch=30, multiplier=1e-1),
]

if hvd_rank == 0:
    verbose = 2
    time_callback = utils.TimeHistory()
    callbacks.append(time_callback)
else:
    verbose = 0

train_steps = int(num_train/BATCH_SIZE/hvd_size)
valid_steps = int(num_valid/BATCH_SIZE*2/hvd_size)

train_start = time.time()

model.fit(train, steps_per_epoch=train_steps, validation_freq=2, 
          validation_data=valid, validation_steps=valid_steps,
          epochs=EPOCHS, callbacks=callbacks, verbose=verbose)

train_end = time.time()

if hvd_rank == 0:
    duration = min(time_callback.times)
    fps = train_steps*BATCH_SIZE/duration
    print("\nResults:\n")
    print("ResNet FPS:")
    print("*", hvd_size, "GPU:", int(fps*hvd_size))
    print("* Per GPU:", int(fps))
    print("Total train time:", int(train_end-train_start))

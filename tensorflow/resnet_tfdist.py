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
parser.add_argument("--batchsize", default=128, type=int,
                    help="Batch size to use for training")
parser.add_argument("--imgsize", default=224, type=int,
                    help="Image size to use for training")
parser.add_argument("--lr", default=0.001, type=float,
                    help="Learning rate")
parser.add_argument("--epochs", default=40, type=int,
                    help="Number of epochs to train for")
parser.add_argument("--stats", action="store_true", default=False,
                    help="Record stats using NVStatsRecorder")
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

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow_datasets as tfds
import cnn_models
import utils

if args.stats:
    from nvstatsrecorder.callbacks import NVStats, NVLinkStats

print("Using XLA:", args.xla)
tf.config.optimizer.set_jit(args.xla)
print("Using grappler AMP:", args.amp)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp})

strategy = tf.distribute.MirroredStrategy()
replicas = strategy.num_replicas_in_sync

LEARNING_RATE = args.lr
BATCH_SIZE = args.batchsize * replicas
IMG_SIZE = (args.imgsize, args.imgsize)
EPOCHS = args.epochs

print("Number of devices:", replicas)
print("Global batch size:", BATCH_SIZE)
print("Adjusted learning rate:", LEARNING_RATE)


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

train = dataset["train"]
train = train.shuffle(100000)
train = train.repeat(count=-1)
train = train.map(format_train_example, num_parallel_calls=int(multiprocessing.cpu_count())-1)
train = train.batch(BATCH_SIZE, drop_remainder=True)
train = train.prefetch(128)

valid = dataset["validation"]
valid = valid.repeat(count=-1)
valid = valid.map(format_test_example, num_parallel_calls=int(multiprocessing.cpu_count())-1)
valid = valid.batch(BATCH_SIZE, drop_remainder=False)
valid = valid.prefetch(128)

print("Output:", str(train.take(1)), str(valid.take(1)))

print("Build and distribute model")

if args.keras_amp:
    print("Using Keras AMP:", args.keras_amp)
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    
with strategy.scope():
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
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"])

print("Train model")

verbose = 1
time_callback = utils.TimeHistory()
callbacks = [time_callback]

if args.stats:
    SUDO_PASSWORD = os.environ["SUDO_PASSWORD"]
    nv_stats = NVStats(gpu_index=0)
    nvlink_stats = NVLinkStats(SUDO_PASSWORD, gpus=[0,1,2,3])
    callbacks.append(nv_stats)
    callbacks.append(nvlink_stats)

train_steps = int(num_train/BATCH_SIZE)
valid_steps = int(num_valid/BATCH_SIZE)

train_start = time.time()

with strategy.scope():
    model.fit(train, steps_per_epoch=train_steps, validation_freq=2, 
              validation_data=valid, validation_steps=valid_steps,
              epochs=EPOCHS, callbacks=callbacks, verbose=verbose)
    
train_end = time.time()

if args.stats:
    nv_stats_recorder = nv_stats.recorder
    nvlink_stats_recorder = nvlink_stats.recorder
    nv_stats_recorder.plot_gpu_util(smooth=10, outpath="resnet_gpu_util.png")
    nvlink_stats_recorder.plot_nvlink_traffic(smooth=10, outpath="resnet_nvlink_util.png")

duration = min(time_callback.times)
fps = train_steps*BATCH_SIZE/duration

print("\nResults:\n")
print("ResNet FPS:")
print("*", replicas, "GPU:", int(fps))
print("* Per GPU:", int(fps/replicas))
print("Total train time:", int(train_end-train_start))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rn152", action="store_true", default=False,
                    help="Train a larger ResNet-152 model instead of ResNet-50")
parser.add_argument("--dn201", action="store_true", default=False,
                    help="Train a larger DenseNet-201 model instead of ResNet-50")
parser.add_argument("--mobilenet", action="store_true", default=False,
                    help="Train a smaller MobileNetV2 model instead of ResNet-50")
parser.add_argument("--toycnn", action="store_true", default=False,
                    help="Train a Toy CNN model instead of ResNet-50")
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
parser.add_argument("--epochs", default=90, type=int,
                    help="Number of epochs to train for")
parser.add_argument("--stats", action="store_true", default=False,
                    help="Record stats using NVStatsRecorder")
parser.add_argument("--imagenet2012", action="store_true", default=False,
                    help="Train on ImageNet2012")
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--no_val", action="store_true", default=False)
parser.add_argument("--img_aug", action="store_true", default=False)
parser.add_argument("--ctl", action="store_true", default=False)
args = parser.parse_args()

import os
import multiprocessing
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARN)
import time
import horovod.tensorflow.keras as hvd
import horovod.tensorflow as hvd_tf

hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()
n_cores = multiprocessing.cpu_count()
worker_threads = n_cores//hvd_size

import tensorflow as tf

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = str(worker_threads)

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[hvd_rank], "GPU")
tf.config.experimental.set_memory_growth(gpus[hvd_rank], True)

import tensorflow_datasets as tfds
import cnn_models
import utils
import optimizers
if args.stats:
    from nvstatsrecorder.callbacks import NVStats, NVLinkStats

LEARNING_RATE = args.lr
BATCH_SIZE = args.batchsize
IMG_SIZE = args.imgsize
IMG_SIZE_C = (args.imgsize, args.imgsize, 3)
L_IMG_SIZE = int(args.imgsize*1.2)
EPOCHS = args.epochs

print("Using XLA:", args.xla)
tf.config.optimizer.set_jit(args.xla)
print("Using grappler AMP:", args.amp)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.amp})
tf.config.threading.set_inter_op_parallelism_threads(worker_threads)

print("Loading Dataset")

options = tf.data.Options()
read_config = tfds.ReadConfig(options=options, interleave_parallel_reads=worker_threads)

if args.imagenet2012:
    dataset, info = tfds.load("imagenet2012",
                              read_config=read_config,
                              decoders={'image': tfds.decode.SkipDecoding(),},
                              with_info=True,
                              as_supervised=True)
else:
    dataset, info = tfds.load("imagenette/320px",
                              read_config=read_config,
                              decoders={'image': tfds.decode.SkipDecoding(),},
                              with_info=True,
                              as_supervised=True)
num_class = info.features["label"].num_classes
print("Classes:", num_class)
    
num_train = info.splits["train"].num_examples
num_valid = info.splits["validation"].num_examples

print("Number of training examples:", num_train)
print("Number of validation examples:", num_valid)

if args.img_aug:
    @tf.function
    def format_train_example(_image, label):
        image = tf.io.decode_jpeg(_image, channels=3,
                                  fancy_upscaling=False,
                                  dct_method="INTEGER_FAST")
        image = tf.image.resize_with_pad(image, L_IMG_SIZE, L_IMG_SIZE)
        image = tf.image.random_crop(image, IMG_SIZE_C)
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_class)
        return image, label
else:
    @tf.function
    def format_train_example(_image, label):
        image = tf.io.decode_jpeg(_image, channels=3,
                                  fancy_upscaling=False,
                                  dct_method="INTEGER_FAST")
        image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_class)
        return image, label


@tf.function
def format_test_example(_image, label):
    image = tf.io.decode_jpeg(_image, channels=3,
                              fancy_upscaling=False,
                              dct_method="INTEGER_FAST")
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_class)
    return image, label

print("Build tf.data input pipeline")

train = dataset["train"].shard(num_shards=hvd_size, index=hvd_rank)
train.options().experimental_threading.private_threadpool_size = worker_threads
train = train.shuffle(16384)
train = train.repeat(count=-1)
train = train.map(format_train_example, num_parallel_calls=worker_threads)
train = train.batch(BATCH_SIZE, drop_remainder=True)
train = train.prefetch(50)
print("Running pipeline:")
for batch in train.take(1):
    print("* Image shape:", tf.shape(batch[0]))
    _ = str(batch[0].numpy()).replace("\n", " ")
    print("* Label shape:", tf.shape(batch[1]))
time.sleep(1)

valid = dataset["validation"].shard(num_shards=hvd_size, index=hvd_rank)
valid = valid.repeat(count=-1)
valid = valid.map(format_test_example, num_parallel_calls=worker_threads)
if args.imagenet2012:
    valid = valid.batch(BATCH_SIZE, drop_remainder=False)
else:
    valid = valid.batch(32, drop_remainder=False)
valid = valid.prefetch(50)
print("Running pipeline:")
for batch in valid.take(1):
    print("* Image shape:", tf.shape(batch[0]))
    _ = str(batch[0].numpy())
    print("* Label shape:", tf.shape(batch[1]))
time.sleep(1)
    
time.sleep(1)

print("Build and distribute model")

if hvd_rank == 0:
    print("Number of devices:", hvd_size)
    print("Global batch size:", BATCH_SIZE*hvd_size)
    print("Adjusted learning rate:", LEARNING_RATE)

if args.keras_amp:
    print("Using Keras AMP:", args.keras_amp)
    tf.keras.mixed_precision.experimental.set_policy("mixed_float16")
    
if args.rn152:
    print("Using ResNet-152 model")
    model = cnn_models.rn152((IMG_SIZE,IMG_SIZE), num_class, weights=None)
elif args.dn201:
    print("Using DenseNet-201 model")
    model = cnn_models.dn201((IMG_SIZE,IMG_SIZE), num_class, weights=None)
elif args.mobilenet:
    print("Using MobileNetV2 model")
    model = cnn_models.mobilenet((IMG_SIZE,IMG_SIZE), num_class, weights=None)
else:
    print("Using ResNet-50 model")
    model = cnn_models.rn50((IMG_SIZE,IMG_SIZE), num_class, weights=None)
if args.fp16comp:
    print("Using float16 compression for all-reduce")
    compression = hvd.Compression.fp16
else:
    compression = hvd.Compression.none
opt = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.8)
#opt = optimizers.NovoGrad(lr=LEARNING_RATE)
if not args.ctl:
    if args.amp:
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
    opt = hvd.DistributedOptimizer(opt, compression=compression)
loss = tf.keras.losses.CategoricalCrossentropy()
if args.ctl:
    print("Not needed to compile model for custom training loop")
    model.compile(loss=loss,
                  optimizer=opt)
else:
    print("Using Keras model.fit()")
    print("* Cannot use experimental_run_tf_function")
    print("* Using Keras accuracy metric")
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[tf.keras.metrics.Accuracy()],
                  experimental_run_tf_function=False)
try:
    model.load_weights("checkpoint.h5")
except Exception as e:
    print(e)
    print("Not resuming from checkpoint")

print("Train model")

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=1, verbose=1),
    hvd.callbacks.LearningRateScheduleCallback(start_epoch=1, end_epoch=90, multiplier=1.),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3)
]

if hvd_rank == 0:
    verbose = 1
    time_callback = utils.TimeHistory()
    checkpoints = tf.keras.callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    callbacks.append(time_callback)
    callbacks.append(checkpoints)
    if args.stats:
        SUDO_PASSWORD = os.environ["SUDO_PASSWORD"]
        nv_stats = NVStats(gpu_index=0, interval=5)
        nvlink_stats = NVLinkStats(SUDO_PASSWORD, gpus=[0,1,2,3], interval=5)
        callbacks.append(nv_stats)
        callbacks.append(nvlink_stats)
else:
    verbose = 0

if args.imagenet2012:
    train_steps = int(num_train/BATCH_SIZE/hvd_size)
    valid_steps = int(num_valid/BATCH_SIZE/hvd_size)
else:
    train_steps = int(num_train/BATCH_SIZE*2/hvd_size)
    valid_steps = int(num_valid/32/hvd_size)

if args.steps:
    train_steps = args.steps
if args.imagenet2012 and args.steps:
    valid_steps = args.steps

train_start = time.time()

if args.ctl:
    from tqdm import tqdm
    
    @tf.function
    def train_first_step(images, labels):
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            loss_value = loss(labels, probs)
        tape = hvd_tf.DistributedGradientTape(tape, compression=compression)
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        hvd_tf.broadcast_variables(model.variables, root_rank=0)
        hvd_tf.broadcast_variables(opt.variables(), root_rank=0)
        
    @tf.function
    def train_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            probs = model(images, training=True)
            loss_value = loss(labels, probs)
        tape = hvd_tf.DistributedGradientTape(tape, compression=compression)
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value
    
    train_iter = iter(train)
    
    def train_epoch(train_steps):
        for batch in range(train_steps):
            inputs = next(train_iter)
            loss_value = train_step(inputs)
        return True
    
    def train_loop(num_epochs):
        print("Starting custom training loop")
        time_history = []
        print("Do hvd broadcast_variables")
        for batch, (images, labels) in enumerate(train.take(1)):
            train_first_step(images, labels)
        print("Start actual training")
        for epoch in range(num_epochs):
            print("\nEpoch:", epoch+1, "/", num_epochs)
            st = time.time()
            _ = train_epoch(train_steps)
            et = time.time()
            duration = et-st
            print("Epoch", epoch+1, "took",  str(int(duration))+"s")
            time_history.append(duration)
        return time_history
            
    time_history = train_loop(EPOCHS)

else:
    print("Starting model.fit()")
    if args.no_val:
        model.fit(train, steps_per_epoch=train_steps,
                  epochs=EPOCHS, callbacks=callbacks, verbose=verbose)
    else:
        model.fit(train, steps_per_epoch=train_steps, validation_freq=2, 
                  validation_data=valid, validation_steps=valid_steps,
                  epochs=EPOCHS, callbacks=callbacks, verbose=verbose)

train_end = time.time()

if hvd_rank == 0:
    if args.stats:
        nv_stats_recorder = nv_stats.recorder
        nvlink_stats_recorder = nvlink_stats.recorder
        nv_stats_recorder.plot_gpu_util(smooth=5, outpath="resnet_gpu_util.png")
        nvlink_stats_recorder.plot_nvlink_traffic(smooth=5, outpath="resnet_nvlink_util.png")
    if args.ctl:
        duration = min(time_history)
    else:
        duration = min(time_callback.times)
    fps = train_steps*BATCH_SIZE/duration
    print("\n")
    print("Results:")
    print("========\n")
    print("ResNet FPS:")
    print("*", hvd_size, "GPU:", int(fps*hvd_size))
    print("* Per GPU:", int(fps))
    print("Total train time:", int(train_end-train_start))

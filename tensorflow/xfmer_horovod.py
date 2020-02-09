import os
import time
import argparse
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"

import tensorflow.compat.v2 as tf
import horovod.tensorflow.keras as hvd

parser = argparse.ArgumentParser(
    description="Train and evaluate Transformers for various GLUE tasks",
)

parser.add_argument("--amp", action="store_true", default=False,
                    help="Use grappler AMP for mixed precision training")
parser.add_argument("--keras_amp", action="store_true", default=False,
                    help="Use Keras AMP for mixed precision training")
parser.add_argument("--xla", action="store_true", default=False,
                    help="Use XLA compiler")
parser.add_argument("--fp16comp", action="store_true", default=True,
                    help="Use float16 compression during allreduce")
parser.add_argument("--epochs", default=10,
                    help="Number of epochs to train for",
                    type=int)
parser.add_argument("--batch_size", default=6,
                    help="Batch size to use for training",
                    type=int)
parser.add_argument("--lr", default=1e-5,
                    help="Learning Rate to use for training",
                    type=float)
parser.add_argument("--maxseqlen", default=128,
                    help="Maximum input sequence length",
                    type=int)
parser.add_argument("--task", default="mrpc",
                    help="Task for training and evaluation")
parser.add_argument("--model", default="xlm-mlm-en-2048",
                    help="Which Transformer model to use")
parser.add_argument("--outpath", default="./output/",
                    help="Output path")

args = parser.parse_args()

LEARNING_RATE = float(args.lr)
USE_XLA = args.xla
USE_AMP = args.amp
model_name = args.model
MAX_SEQ_LEN = args.maxseqlen
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

if args.outpath:
    OUT_PATH = args.outpath
    if OUT_PATH[-1] != "/":
        OUT_PATH = OUT_PATH + "/"
else:
    OUT_PATH = args.outpath
os.makedirs(OUT_PATH, exist_ok=True)

hvd.init()
hvd_rank = hvd.local_rank()
hvd_size = hvd.size()

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(physical_devices[hvd_rank], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[hvd_rank], True)

import transformers as xfmers
import xfmer_utils

# training parameters

task_list = {
    "cola": "glue/cola",
    "mrpc": "glue/mrpc",
    "sst-2": "glue/sst2",
}

task_name = args.task
if task_name == "sst2":
    task_name = "sst-2"
dataset_name = task_list[task_name]

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({
    "auto_mixed_precision": USE_AMP
})

print("Building input pipeline...")

tokenizer = xfmer_utils.create_tokenizer(model_name)
task_dataset = xfmer_utils.return_glue_task(tokenizer, dataset_name, task_name, MAX_SEQ_LEN)
train_dataset = task_dataset["train_dataset"]
train_dataset = train_dataset.shard(hvd_size, hvd_rank)
train_dataset = train_dataset.repeat().shuffle(100000).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(64)

valid_dataset = task_dataset["valid_dataset"].repeat().batch(BATCH_SIZE)
valid_dataset = valid_dataset.prefetch(64)

print(hvd_rank, "Building model...")

model = xfmer_utils.create_model(model_name, task_dataset["num_labels"])
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
if args.fp16comp:
    print("Using float16 compression for all-reduce")
    compression = hvd.Compression.fp16
else:
    compression = hvd.Compression.none
opt = hvd.DistributedOptimizer(opt,
                               compression=compression,
                               sparse_as_dense=True)
if USE_AMP:
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=opt,
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
              experimental_run_tf_function=False)

# train model
    
train_steps_per_epoch = int(task_dataset["train_examples"]/BATCH_SIZE/hvd_size)
valid_steps_per_epoch = int(task_dataset["valid_examples"]/BATCH_SIZE/hvd_size)

lr_schedule = hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=1, steps_per_epoch=train_steps_per_epoch)
time_callback = xfmer_utils.TimeHistory()
hvd_broadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
hvd_metric = hvd.callbacks.MetricAverageCallback()

callbacks_list = [hvd_broadcast, hvd_metric, lr_schedule]

if hvd_rank == 0:
    if USE_XLA:
        print("XLA is enabled. First run will be delayed due to XLA JIT compilation.")
    if USE_AMP:
        print("Model is using Automatic Mixed Precision")
    verbose = 2
    model.summary()
    callbacks_list.append(time_callback)
else:
    verbose = 0
    
print(hvd_rank, "Starting training...")

log = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=train_steps_per_epoch,
                callbacks=callbacks_list, verbose=verbose)

if hvd_rank == 0:    
    # results
    
    cold_start_duration = max(time_callback.times)
    epoch_duration = min(time_callback.times)
    eg_per_sec = int(train_steps_per_epoch*BATCH_SIZE/epoch_duration)

    print("\n\n=================\n\n")
    print("\nResults (DIST/XLA/AMP:", "horovod", USE_XLA, USE_AMP, ")", task_name)
    print("Total time:", int(sum(time_callback.times)), "seconds")
    print("Cold Start time:", int(cold_start_duration - epoch_duration))
    print("Training Throughput:", eg_per_sec * hvd_size, "examples per second")

    print("Throughput per GPU:", eg_per_sec, "examples per second")

    score = model.evaluate(valid_dataset, steps=valid_steps_per_epoch)
    print("Loss:", score[0])
    print("Accuracy:", score[1])
    print("\n\n=================\n\n")

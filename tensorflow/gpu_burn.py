import os
import multiprocessing
n_threads = multiprocessing.cpu_count()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = str(n_threads)

import time
from tqdm import tqdm
import tensorflow as tf
tf.config.experimental.set_synchronous_execution(False)
gpu_list = tf.config.list_physical_devices("GPU")
print("GPUs:", gpu_list)
num_gpus = len(gpu_list)
for gpu in gpu_list:
    tf.config.experimental.set_memory_growth(gpu, False)
    
from nvstatsrecorder.recorders import NVStatsRecorder, NVLinkStatsRecorder

@tf.function(experimental_compile=True)
def do_op(a, b):
    return tf.linalg.matmul(a, b)

def benchmark_matmul(M, gpus=1, dtype=tf.float32, iterations=100):
    # generate data and warm-up iteration
    slots = []
    for i in range(num_gpus):
        with tf.device("/GPU:"+str(i)):
            A = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
            B = tf.random.normal([M, M], mean=0, stddev=1, dtype=dtype)
            C = do_op(A, B)
            slots.append((A, B))
    C.numpy()
    # measure overhead
    st = time.time()
    for _ in range(1):
        for i in range(num_gpus):
            with tf.device("/GPU:"+str(i)):
                C = do_op(slots[i][0], slots[i][1])
    C.numpy()
    et = time.time()
    overhead = et - st
    # run benchmark
    st = time.time()
    for _ in range(iterations+1):
        for i in range(num_gpus):
            with tf.device("/GPU:"+str(i)):
                C = do_op(slots[i][0], slots[i][1])
    C.numpy()
    et = time.time()
    duration = (et-st) - overhead
    return gpus*iterations/duration

fp16_matmul, fp32_matmul, fp64_matmul = [], [], []
fp16_tflops, fp32_tflops, fp64_tflops = [], [], []

M_list = [16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 4, 2, 1]

nv_stats_recorder = NVStatsRecorder(gpu_index=0)
nv_stats_recorder.start(interval=1)

print("\nStarting burn...\n")

iterations = 40

for M in tqdm(M_list):
    ret = benchmark_matmul(M, gpus=num_gpus, dtype=tf.float64, iterations=iterations)
    tflops = ret * 2 * M**3 / 1e12
    fp64_matmul.append(ret)
    fp64_tflops.append(tflops)
    time.sleep(1)
    ret = benchmark_matmul(M, gpus=num_gpus, dtype=tf.float32, iterations=iterations)
    tflops = ret * 2 * M**3 / 1e12
    fp32_matmul.append(ret)
    fp32_tflops.append(tflops)
    time.sleep(1)
    ret = benchmark_matmul(M, gpus=num_gpus, dtype=tf.float16, iterations=iterations)
    tflops = ret * 2 * M**3 / 1e12
    fp16_matmul.append(ret)
    fp16_tflops.append(tflops)
    time.sleep(1)
    
nv_stats_recorder.stop()

print("\nFinished!\n")

nv_stats_recorder.plot_gpu_util(smooth=3, outpath="burn_gpu_util.png")
    
title = "Maximum TFLOPS achieved on " + str(num_gpus) + " GPUs"
print("")
print(title)
print("="*len(title))
print("* FP64:", int(max(fp64_tflops)), "TFLOPS")
print("* FP32:", int(max(fp32_tflops)), "TFLOPS")
print("* FP16:", int(max(fp16_tflops)), "TFLOPS")
print("")

from matplotlib import pyplot as plt
plt.clf()
plt.figure(figsize=(10,6), dpi=100)
plt.title("Achieved TFLOPS from MatMul")
plt.plot(M_list, fp16_tflops, label="FP16", color="g")
plt.plot(M_list, fp32_tflops, label="FP32", color="b")
plt.plot(M_list, fp64_tflops, label="FP64", color="r")
plt.xlabel("Matrix size M*M")
plt.ylabel("Achieved TFLOPS")
plt.legend()
plt.show()
plt.savefig("tflops_plot.png")

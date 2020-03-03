## Image Classification on Imagenette (smaller ImageNet)

### Test Case 1 ("small CNN")

* Model: ResNet-50
* Framework: TF2.1 + tf.distribute
* Uses NCCL: Yes

```shell
python3 resnet_tfdist.py --amp --xla
```

| V100 | Training time | Images/sec | Val Acc |
| ---- | ------------- | ---------- | ------- |
| 4    | 280s          | 4161       | 0.8208  |

**DenseNet-201 + Horovod + OpenMPI + NCCL**

### Test Case 2 ("big CNN")

* Model: DenseNet-201
* Framework: TF2.1 + Horovod + OpenMPI
* Uses NCCL: Yes

```shell
mpirun -np 4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 resnet_horovod.py --amp --xla --dn201 --imgsize 256 --batchsize 56
```

| V100 | Training time | Images/sec | Val Acc |
| ---- | ------------- | ---------- | ------- |
| 4    | 1056s          | 750       | 0.9241  |

## Transformer Fine-tuning

WIP

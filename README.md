# GradCosine

## Installation
There is no need to install the environment. Instead, what the users need to do is to pull the docker from the internet.
```commandline
docker pull harbory/hycls:latest
```

If you cannot access the Internet in your environment, you can build the docker image by yourself (switch to docker_env folder first).
```commandline
docker build --network=host -t hycls:latest .
```

If your environment does not support docker, please refer to the Dockerfile for details about the dependencies.

## Datasets Preparation
Please prepare imagenet-1k datasets and put it into /path/to/data/imagenet.
```text
imagenet
├── train
│   ├── n01440764
│   │   ├── n01440764_18.JPEG
│   │   ├── ...
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ...
├── meta
│   ├── train.txt
│   ├── val.txt
```


## Getting Start
Please create a docker container and enter it:
```commandline
DATALOC=/path/to/data LOGLOC=/path/to/logger bash tools/docker.sh
```


To perform gradcos to init the neural network.
```commandline
PYTHONPATH=. python tools/gradcos_runner.py configs/mmcls/resnet_gradcos_init/resnet50_gc_gn.py --seed 0 --deterministic --work-dir ./work_dir/init
```

To train the initialized network.
```commandline
bash tools/dist_train.sh configs/mmcls/resnet/resnet50_8xb32_in1k.py 8 --seed 0 --deterministic --work-dir work_dir/train --load-from work_dir/init/init.pth
```

Since we adopt the deterministic training, you will get exactly the same results as in the paper if everything works well. I have tested the above scripts so that you can find the [gradcos](logs/init.txt) and [training](logs/train.txt) logs.
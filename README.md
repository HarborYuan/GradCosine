# GradCosine

## Resnet-50
```commandline
PYTHONPATH=. python tools/gradcos_runner.py configs/mmcls/resnet_gradcos_init/resnet50_gc_gn.py --seed 0 --deterministic --work-dir ./work_dir/init && bash tools/dist_train.sh configs/mmcls/resnet/resnet50_8xb32_in1k.py 8 --seed 0 --deterministic --work-dir work_dir/train --load-from work_dir/init/init.pth
```
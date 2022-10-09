FROM nvcr.io/nvidia/pytorch:21.10-py3

# install ujson
RUN pip install ujson

# handle the timezone
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="PRC" apt-get install tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && unlink /etc/localtime && ln -s /usr/share/zoneinfo/PRC /etc/localtime

# mmcv : 1.4.8
RUN until MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@25602c68441d18430d649711a4f3bec576ce327e; do sleep 0.1; done

# mmdet : 2.23.0
RUN until FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmdetection.git@3e2693151add9b5d6db99b944da020cba837266b; do sleep 0.1; done

# panopticapi
RUN until python -m pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0; do sleep 0.1; done

# timm : 2204
RUN until python -m pip install git+https://github.com/rwightman/pytorch-image-models.git@01a0e25a67305b94ea767083f4113ff002e4435c; do sleep 0.1; done

# mmseg : 0.23.0
RUN until python -m pip install git+https://github.com/open-mmlab/mmsegmentation.git@6670672345200cd6e855a2848d38cb56c9a15e82; do sleep 0.1; done

# mmcls 0.22.0
RUN until python -m pip install git+https://github.com/open-mmlab/mmclassification.git@2e7f99946b899a7ae4eaec47e87b1a8d5d7cffe4; do sleep 0.1; done

WORKDIR /opt/project/


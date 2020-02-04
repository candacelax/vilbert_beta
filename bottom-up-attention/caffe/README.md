# Caffe

## Docker
Install cudnn (see [instructions](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#download)).
Change Python version in CMakeLists.txt, line 34
Set your CUDA version in Dockerfile, find list of tags [here](https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md). Be sure it's the "devel" version.
Install nvidia-container-runtime, (see [instructions](https://github.com/NVIDIA/nvidia-container-runtime)).

# TODO changes in cmake/Cuda.cmake, CMakeLists.txt
``` bash    
    sudo docker build -f caffe/docker/standalone/gpu/Dockerfile -t caffe_image_features .
    sudo docker container run -t -v /storage/ccross/bias-grounded-bert/vilbert_beta/bottom-up-attention/caffe/features/conceptual:/opt/features/conceptual --gpus all caffe_image_features python2.7 /opt/tools/generate_tsv.py --cfg /opt/experiments/cfgs/faster_rcnn_end2end_resnet.yml --def /opt/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /opt/features/conceptual/conceptual_resnet101_faster_rcnn_genome.tsv --net /opt/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --total_group 1 --group_id 0 --split conceptual_image_val --data_dir /opt/data/conceptual-captions --gpu 0,1,2,3,4,5,6,7

    sudo docker container run -t -v /storage/ccross/bias-grounded-bert/vilbert_beta/bottom-up-attention/caffe/features/conceptual:/opt/features/conceptual --gpus all caffe_image_features python2.7 /opt/tools/generate_tsv.py --cfg /opt/experiments/cfgs/faster_rcnn_end2end_resnet.yml --def /opt/models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out /opt/features/google-images/weat3_resnet101_faster_rcnn_genome.tsv --net /opt/data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --total_group 1 --group_id 0 --split google_images --data_dir /opt/data/google-images/weat3 --gpu 0,1,2,3,4,5,6,7


    sudo docker create -ti --name dummy caffe_image_features bash
    sudo docker cp dummy:/opt/features/conceptual/conceptual_resnet101_faster_rcnn_genome.tsv.0 check_features.tsv
    sudo docker rm -f dummy
```


[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

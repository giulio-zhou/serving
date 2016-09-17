#!/usr/bin/env bash

# ./bazel-bin/tensorflow_serving/example/no_rpc_benchmarker \
#   --mnist_file=/crankshaw-local/mnist/data/test.data \
#   --num_requests=1000 --num_batch_threads=2 \
#   /crankshaw-local/tf_models/mnist_from_guilio/mnist_convnet/tf_serving_model/convnet
./bazel-bin/tensorflow_serving/example/no_rpc_cifar10_benchmarker \
  --mnist_file=/giulio-local/cifar10_data/cifar10.txt \
  --num_requests=100000 \
  --num_batch_threads=2 \
  /giulio-local/tf_net_models/cifar10_convnet/tf_serving_model/
  # /crankshaw-local/tf_models/tfserve/mnist_conv_b64/


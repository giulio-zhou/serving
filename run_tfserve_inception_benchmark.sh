#!/usr/bin/env bash

# ./bazel-bin/tensorflow_serving/example/no_rpc_benchmarker \
#   --mnist_file=/crankshaw-local/mnist/data/test.data \
#   --num_requests=1000 --num_batch_threads=2 \
#   /crankshaw-local/tf_models/mnist_from_guilio/mnist_convnet/tf_serving_model/convnet
./bazel-bin/tensorflow_serving/example/no_rpc_inception_benchmarker \
  --inception_data_file=/giulio-local/flowers-data/raw-data/train/inception_data.txt \
  --num_requests=5000 \
  --num_batch_threads=2 \
  --batch_size=16 \
  /giulio-local/tf_net_models/inception_net/tf_serving_model/
  # /crankshaw-local/tf_models/tfserve/mnist_conv_b64/


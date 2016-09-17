#!/usr/bin/env bash

# ./bazel-bin/tensorflow_serving/example/no_rpc_benchmarker \
#   --mnist_file=/crankshaw-local/mnist/data/test.data \
#   --num_requests=1000 --num_batch_threads=2 \
#   /crankshaw-local/tf_models/mnist_from_guilio/mnist_convnet/tf_serving_model/convnet
./bazel-bin/tensorflow_serving/example/no_rpc_benchmarker \
  --mnist_file=/crankshaw-local/mnist/data/test.data \
  --num_requests=1000000 \
  --num_batch_threads=1 \
  /giulio-local/tf_net_models/mnist_convnet/tf_serving_model/convnet/
  # /crankshaw-local/tf_models/tfserve/mnist_conv_b64/


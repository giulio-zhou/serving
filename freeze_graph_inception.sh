./bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=/giulio-local/clipper/model_wrappers/python/mnist.txt \
						 --input_checkpoint=/giulio-local/tf_net_models/mnist_convnet/tf_checkpoint/convnet.ckpt \
						 --output_node_names=Softmax \
						 --output_graph=/giulio-local/tf_net_models/mnist_convnet/tf_checkpoint/mnist.pb

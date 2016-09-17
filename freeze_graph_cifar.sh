./bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=/giulio-local/clipper/model_wrappers/python/cifar10.txt \
						 --input_checkpoint=/giulio-local/tf_net_models/cifar10_convnet/tf_checkpoint/model.ckpt-19999 \
						 --output_node_names=Softmax \
						 --output_graph=/giulio-local/tf_net_models/cifar10_convnet/tf_checkpoint/cifar10.pb

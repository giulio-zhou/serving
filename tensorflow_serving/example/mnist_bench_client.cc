
#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>


#include "grpc++/channel.h"
#include "grpc++/client_context.h"
#include "grpc++/create_channel.h"
#include "grpc++/security/credentials.h"


// #include "grpc++/completion_queue.h"
// #include "grpc++/security/server_credentials.h"
// #include "grpc++/server.h"
// #include "grpc++/server_builder.h"
// #include "grpc++/server_context.h"
// #include "grpc++/support/async_unary_call.h"
// #include "grpc++/support/status.h"
// #include "grpc++/support/status_code_enum.h"
#include "grpc/grpc.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/batching/basic_batch_scheduler.h"
#include "tensorflow_serving/batching/batch_scheduler.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/example/mnist_inference.grpc.pb.h"
#include "tensorflow_serving/example/mnist_inference.pb.h"
#include "tensorflow_serving/servables/tensorflow/simple_servers.h"
#include "tensorflow_serving/session_bundle/manifest.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/session_bundle/signature.h"
#include "tensorflow_serving/util/threadpool_executor.h"



using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;


// using grpc::InsecureServerCredentials;
// using grpc::Server;
// using grpc::ServerAsyncResponseWriter;
// using grpc::ServerBuilder;
// using grpc::ServerContext;
// using grpc::ServerCompletionQueue;
// using grpc::Status;
using grpc::StatusCode;
using tensorflow::serving::MnistRequest;
using tensorflow::serving::MnistResponse;
using tensorflow::serving::MnistService;
using tensorflow::string;

namespace {

  

  int RunBenchmark(int numtests) {
    std::shared_ptr<Channel> channel = grpc::CreateChannel("localhost:9001",
                                  grpc::InsecureChannelCredentials());
    std::unique_ptr<MnistService::Stub> stub_ = MnistService::NewStub(channel);
    ClientContext context;
    MnistRequest request;
    request.image_data.
    stub_->Classify(&context, 

  }

} // namespace

int main(int argc, char** argv) {
  // Parse command-line options.
  tensorflow::int32 port = 0;
  const bool parse_result =
      tensorflow::ParseFlags(&argc, argv, {tensorflow::Flag("port", &port)});
  if (!parse_result) {
    LOG(FATAL) << "Error parsing command line flags.";
  }
  if (argc != 2) {
    LOG(FATAL) << "Usage: mnist_inference_2 --port=9000 /path/to/exports";
  }
  const string export_base_path(argv[1]);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  // WARNING(break-tutorial-inline-code): The following code snippet is
  // in-lined in tutorials, please update tutorial documents accordingly
  // whenever code changes.
  std::unique_ptr<tensorflow::serving::Manager> manager;
  tensorflow::Status status = tensorflow::serving::simple_servers::
      CreateSingleTFModelManagerFromBasePath(export_base_path, &manager);

  TF_CHECK_OK(status) << "Error creating manager";

  // Wait until at least one model is loaded.
  std::vector<tensorflow::serving::ServableId> ready_ids;
  // TODO(b/25545573): Create a more streamlined startup mechanism than polling.
  do {
    LOG(INFO) << "Waiting for models to be loaded...";
    tensorflow::Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /*1 sec*/);
    ready_ids = manager->ListAvailableServableIds();
  } while (ready_ids.empty());

  // Run the service.
  RunServer(port, ready_ids[0].name, std::move(manager));

  return 0;
}

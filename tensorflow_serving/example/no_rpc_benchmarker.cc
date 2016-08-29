/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A gRPC server that classifies images into digit 0-9.
// Given each request with an image pixels encoded as floats, the server
// responds with 10 float values as probabilities for digit 0-9 respectively.
// The classification is done by running image data through a convolutional
// network trained and exported by mnist_model.py.
// The server constantly monitors a file system storage path for models.
// Whenever a new version of model is available, it eagerly unloads older
// version before loading the new one. The server also batches multiple
// requests together and does batched inference for efficiency.
// The intention of this example to demonstrate usage of DymanicManager,
// VersionPolicy and BasicBatchScheduler.

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <random>

#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
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

// using grpc::InsecureServerCredentials;
// using grpc::Server;
// using grpc::ServerAsyncResponseWriter;
// using grpc::ServerBuilder;
// using grpc::ServerContext;
// using grpc::ServerCompletionQueue;
// using grpc::Status;
// using grpc::StatusCode;
using tensorflow::serving::MnistRequest;
using tensorflow::serving::MnistResponse;
// using tensorflow::serving::MnistService;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::serving::ClassificationSignature;

// namespace {
const int kImageSize = 28;
const int kNumChannels = 1;
const int kImageDataSize = kImageSize * kImageSize * kNumChannels;
const int kNumLabels = 10;
const int kEvalBatchSize = 64;

// // A Task holds all of the information for a single inference request.
struct Task : public tensorflow::serving::BatchTask {
  ~Task() override = default;
  size_t size() const override { return 1; }

  Task(std::atomic_ulong& pred_counter, MnistRequest& request)
      : pred_counter(pred_counter), request(request) {}

  std::atomic_ulong& pred_counter;
  MnistRequest& request;
};


void Classify(
    std::unique_ptr<tensorflow::serving::BasicBatchScheduler<Task>>* batch_scheduler,
    std::atomic_ulong& pred_counter, MnistRequest& request) {
  // Verify input.
  if (request.image_data_size() != kImageDataSize) {
     LOG(WARNING) << tensorflow::strings::StrCat(
                   "expected image_data of size ", kImageDataSize,
                   ", got ", request.image_data_size());
    return;
  }

  // LOG(INFO) << "AAAAAAAA";
  // Create and submit a task to the batch scheduler.
  std::unique_ptr<Task> task(new Task(pred_counter, request));
  tensorflow::Status status = (*batch_scheduler)->Schedule(&task);
  // LOG(INFO) << "BBBBBBBBBBB";

  if (!status.ok()) {
    LOG(WARNING) << status.error_message();
    return;
  }
}

// Produces classifications for a batch of requests and associated responses.
void DoClassifyInBatch(
    const string servable_name,
    std::unique_ptr<tensorflow::serving::Manager>* manager,
    std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {

  // LOG(INFO) << "CCCCCCCCCCCCCCC";
  batch->WaitUntilClosed();
  // LOG(INFO) << "DDDDDDDDDDDD";
  if (batch->empty()) {
    return;
  }
  // LOG(INFO) << "EEEEEEEEEEEEEEEE";
  const int batch_size = batch->num_tasks();

  // Replies to each task with the given error status.
  auto complete_with_error = [&batch](const string& msg) {
    LOG(WARNING) << msg;
    // Status status(code, msg);
    // for (int i = 0; i < batch->num_tasks(); i++) {
    //   Task* task = batch->mutable_task(i);
    //   task->calldata->Finish(status);
    // }
  };

  // LOG(INFO) << "FFFFFFFFFFFFF";
  // Get a handle to the SessionBundle.  The handle ensures the Manager does
  // not reload this while it is in use.
  // WARNING(break-tutorial-inline-code): The following code snippet is
  // in-lined in tutorials, please update tutorial documents accordingly
  // whenever code changes.
  auto handle_request =
      tensorflow::serving::ServableRequest::Latest(servable_name);
  tensorflow::serving::ServableHandle<tensorflow::serving::SessionBundle>
      bundle;
  const tensorflow::Status lookup_status =
      (*manager)->GetServableHandle(handle_request, &bundle);
  if (!lookup_status.ok()) {
    complete_with_error(lookup_status.error_message());
    return;
  }

  // LOG(INFO) << "GGGGGGGGGGGGGG";

  // Get the default signature of the graph.  Expected to be a
  // classification signature.
  tensorflow::serving::ClassificationSignature signature;
  const tensorflow::Status signature_status =
      GetClassificationSignature(bundle->meta_graph_def, &signature);
  if (!signature_status.ok()) {
    complete_with_error(signature_status.error_message());
    return;
  }

  // LOG(INFO) << tensorflow::strings::StrCat("Batch size: ", batch_size);
  // Transform protobuf input to inference input tensor.
  // See mnist_model.py for details.
  // WARNING(break-tutorial-inline-code): The following code snippet is
  // in-lined in tutorials, please update tutorial documents accordingly
  // whenever code changes.
  // Tensor input(tensorflow::DT_FLOAT, {batch_size, kImageDataSize});




  Tensor input(tensorflow::DT_FLOAT, {kEvalBatchSize, kImageSize, kImageSize, kNumChannels});
  auto dst = input.flat_outer_dims<float>().data();
  for (int i = 0; i < batch_size; ++i) {
    std::copy_n(
        batch->mutable_task(i)->request.image_data().begin(),
        kImageDataSize, dst);
    dst += kImageDataSize;
  }

  // LOG(INFO) << "IIIIIIIIIIIIIIIIII";

  // // pad to fill out batch size if needed
  for (int i = batch_size; i < kEvalBatchSize; ++i) {
    std::copy_n(
        batch->mutable_task(0)->request.image_data().begin(),
        kImageDataSize, dst);
    // LOG(INFO) << tensorflow::strings::StrCat(
    //         "eeeeeeeeeee: ", i);
    dst += kImageDataSize;
  }

  // LOG(INFO) << "JJJJJJJJJJJJJJJJJJJJJJJ";

  // LOG(INFO) << "created input tensor";

  // Run classification.
  tensorflow::Tensor scores;
  const tensorflow::Status run_status =
      RunClassification(signature, input, bundle->session.get(),
                        nullptr /* classes */, &scores);

  // LOG(INFO) << "KKKKKKKKKKKKKKK";
  if (!run_status.ok()) {
    complete_with_error(run_status.error_message());
    return;
  }
  if (scores.dtype() != tensorflow::DT_FLOAT) {
    complete_with_error(
        tensorflow::strings::StrCat(
            "Expected output Tensor of DT_FLOAT.  Got: ",
            tensorflow::DataType_Name(scores.dtype())));
    return;
  }
  if (scores.dim_size(1) != kNumLabels) {
    complete_with_error(
        tensorflow::strings::StrCat(
            "Expected ", kNumLabels, " labels in each output.  Got: ",
            scores.dim_size(1)));
    return;
  }

  // LOG(INFO) << "LLLLLLLLLLLLLLLLLLL";
  // Transform inference output tensor to protobuf output.
  // See mnist_model.py for details.
  const auto& scores_mat = scores.matrix<float>();
  for (int i = 0; i < batch_size; ++i) {
    batch->mutable_task(i)->pred_counter.fetch_add(1, std::memory_order::memory_order_relaxed);

    // for (int c = 0; c < scores.dim_size(1); ++c) {
    //   calldata->mutable_response()->add_value(scores_mat(i, c));
    // }
    // calldata->Finish(Status::OK);
  }
}


void load_data(const string& path,
    std::unique_ptr<std::vector<MnistRequest>>* data) {

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(path, &file));
  const size_t kBufferSizeBytes = 262144;
  tensorflow::io::InputBuffer in(file.get(), kBufferSizeBytes);
  string line;
  while (in.ReadLine(&line).ok()) {
    std::vector<tensorflow::int32> parsed_line;
    parsed_line.reserve(kImageDataSize + 1);
    bool parse_result = tensorflow::str_util::SplitAndParseAsInts(line, ',', &parsed_line);
    // std::vector<string> cols = str_util::Split(line, ',');
    if (parse_result) {
      int label = parsed_line[0];
      MnistRequest request;
      for (int i = 1; i < kImageDataSize + 1; ++i) {
        request.add_image_data(parsed_line[i] * 1.0 / 255.0);
      }
      (*data)->push_back(request);
    } else {
      LOG(INFO) << "parsing error";
    }
  }

  LOG(INFO) << tensorflow::strings::StrCat("Loaded ", (*data)->size(), " test inputs");
}



int main(int argc, char** argv) {
  // Parse command-line options.
  string mnist_path;
  tensorflow::int32 num_requests;
  tensorflow::int32 num_batch_threads;
  const bool parse_result =
      tensorflow::ParseFlags(&argc, argv, {
          tensorflow::Flag("mnist_file", &mnist_path),
          tensorflow::Flag("num_requests", &num_requests),
          tensorflow::Flag("num_batch_threads", &num_batch_threads),
          });
  if (!parse_result) {
    LOG(FATAL) << "Error parsing command line flags.";
  }
  // LOG(INFO) << tensorflow::strings::StrCat("argc: ", argc);
  // if (argc != 1) {
  //   LOG(FATAL) << "Usage: mnist_inference_2 /path/to/mnist";
  // }
  // const string mnist_path(argv[1]);
  const string servable_path(argv[1]);
  std::unique_ptr<std::vector<MnistRequest>> input_data(new std::vector<MnistRequest>);
  // input_data->reserve(10000);
  load_data(mnist_path, &input_data);
  std::atomic_ulong pred_counter;
  pred_counter.store(0);
  
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::unique_ptr<tensorflow::serving::Manager> manager;
  tensorflow::Status status = tensorflow::serving::simple_servers::
      CreateSingleTFModelManagerFromBasePath(servable_path, &manager);

  TF_CHECK_OK(status) << "Error creating manager";

  // Wait until at least one model is loaded.
  std::vector<tensorflow::serving::ServableId> ready_ids;
  // TODO(b/25545573): Create a more streamlined startup mechanism than polling.
  do {
    LOG(INFO) << "Waiting for models to be loaded...";
    tensorflow::Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /*1 sec*/);
    ready_ids = manager->ListAvailableServableIds();
  } while (ready_ids.empty());

  // Manager in charge of loading and unloading servables.
  // std::unique_ptr<tensorflow::serving::Manager> manager;
  // A scheduler for batching multiple request calls into single calls to
  // Session->Run().
  std::unique_ptr<tensorflow::serving::BasicBatchScheduler<Task>>
      batch_scheduler;

  tensorflow::serving::BasicBatchScheduler<Task>::Options scheduler_options;
  scheduler_options.thread_pool_name = "mnist_service_batch_threads";
  scheduler_options.num_batch_threads = num_batch_threads;
  scheduler_options.batch_timeout_micros = 0;
  scheduler_options.max_batch_size = kEvalBatchSize;
  // scheduler_options.max_batch_size = 1000;
  // Use a very large queue, to avoid rejecting requests. (Note: a production
  // server with load balancing may want to use the default, much smaller,
  // value.)
  scheduler_options.max_enqueued_batches = INT_MAX;
  TF_CHECK_OK(tensorflow::serving::BasicBatchScheduler<Task>::Create(
      scheduler_options,
      [&] (std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
        DoClassifyInBatch(ready_ids[0].name, &manager, std::move(batch));
      },
      &batch_scheduler));

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> mnist_dist(0,input_data->size());

  // std::cout << dist6(rng) << std::endl;
  
  tensorflow::uint64 start_time = tensorflow::Env::Default()->NowMicros();
  for (int i = 0; i < num_requests; ++i) {
    Classify(&batch_scheduler, pred_counter, (*input_data)[mnist_dist(rng)]);
  }
  // LOG(INFO) << "XXXXXXXXXXXXXXXXXXXXXXXX";
  
  while (batch_scheduler->NumEnqueuedTasks() > 0) {
    tensorflow::Env::Default()->SleepForMicroseconds(10 * 1000 /*0.5 sec*/);
  }

  tensorflow::uint64 end_time = tensorflow::Env::Default()->NowMicros();
  tensorflow::uint64 processed_reqs = pred_counter.load();
  double throughput = ((double) processed_reqs) / ((double) (end_time - start_time)) * 1000.0 * 1000.0;
  LOG(INFO) << tensorflow::strings::StrCat("Processed ", processed_reqs,
      " predictions in ", (double) (end_time - start_time) / 1000.0, "ms. Throughput: ",
      throughput);
  return 0;
}

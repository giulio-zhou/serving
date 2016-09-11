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

// A gRPC server that serves inception model exported by inception_export.py.
// Given each request with an image as JPEG encoded byte stream, the server
// responds with kNumTopClasses integer values as indexes of top k matched
// categories and kNumTopClasses float values as the corresponding
// probabilities.

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <iostream>

// #include "grpc++/completion_queue.h"
// #include "grpc++/security/server_credentials.h"
// #include "grpc++/server.h"
// #include "grpc++/server_builder.h"
// #include "grpc++/server_context.h"
// #include "grpc++/support/async_unary_call.h"
// #include "grpc++/support/status.h"
// #include "grpc++/support/status_code_enum.h"
// #include "grpc/grpc.h"
#include "tensorflow/contrib/session_bundle/manifest.pb.h"
#include "tensorflow/contrib/session_bundle/session_bundle.h"
#include "tensorflow/contrib/session_bundle/signature.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
#include "tensorflow_serving/example/inception_inference.grpc.pb.h"
#include "tensorflow_serving/example/inception_inference.pb.h"
// #include "tensorflow_serving/example/mnist_inference.grpc.pb.h"
// #include "tensorflow_serving/example/mnist_inference.pb.h"
#include "tensorflow_serving/servables/tensorflow/simple_servers.h"

// using grpc::InsecureServerCredentials;
// using grpc::Server;
// using grpc::ServerAsyncResponseWriter;
// using grpc::ServerBuilder;
// using grpc::ServerContext;
// using grpc::ServerCompletionQueue;
// using grpc::Status;
// using grpc::StatusCode;
using tensorflow::serving::InceptionRequest;
using tensorflow::serving::InceptionResponse;
// using tensorflow::serving::InceptionRequest;
// using tensorflow::serving::InceptionResponse;
// using tensorflow::serving::InceptionService;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::serving::ClassificationSignature;

// const int kNumTopClasses = 5;
const int kImageSize = 299;
const int kNumChannels = 3;
const int kImageDataSize = kImageSize * kImageSize * kNumChannels;
// const int kNumLabels = 1000;
const int kNumLabels = 5;
int kEvalBatchSize = 50;

// namespace {
// class InceptionServiceImpl;

// Class encompassing the state and logic needed to serve a request.
/*
class CallData {
 public:
  CallData(InceptionServiceImpl* service_impl,
           InceptionService::AsyncService* service,
           ServerCompletionQueue* cq);

  void Proceed();

  void Finish(Status status);

  const InceptionRequest& request() { return request_; }

  InceptionResponse* mutable_response() { return &response_; }

 private:
  // Service implementation.
  InceptionServiceImpl* service_impl_;

  // The means of communication with the gRPC runtime for an asynchronous
  // server.
  InceptionService::AsyncService* service_;
  // The producer-consumer queue where for asynchronous server notifications.
  ServerCompletionQueue* cq_;
  // Context for the rpc, allowing to tweak aspects of it such as the use
  // of compression, authentication, as well as to send metadata back to the
  // client.
  ServerContext ctx_;

  // What we get from the client.
  InceptionRequest request_;
  // What we send back to the client.
  InceptionResponse response_;

  // The means to get back to the client.
  ServerAsyncResponseWriter<InceptionResponse> responder_;

  // Let's implement a tiny state machine with the following states.
  enum CallStatus { CREATE, PROCESS, FINISH };
  CallStatus status_;  // The current serving state.
}; */

// A Task holds all of the information for a single inference request.
struct Task : public tensorflow::serving::BatchTask {
  ~Task() override = default;
  size_t size() const override { return 1; }

  Task(std::atomic_ulong& pred_counter, std::atomic_ulong& latency_sum_micros, InceptionRequest& request)
      : pred_counter(pred_counter), latency_sum_micros(latency_sum_micros), request(request) {}
  // Task(CallData* calldata_arg)
  //     : calldata(calldata_arg) {}

  // CallData* calldata;
  std::atomic_ulong& pred_counter;
  std::atomic_ulong& latency_sum_micros;
  InceptionRequest& request;
};

/*
class InceptionServiceImpl final {
 public:
  InceptionServiceImpl(const string& servable_name,
                       std::unique_ptr<tensorflow::serving::Manager> manager);

  void Classify(CallData* call_data);

  // Produces classifications for a batch of requests and associated responses.
  void DoClassifyInBatch(
      std::unique_ptr<tensorflow::serving::Batch<Task>> batch);

  // Name of the servable to use for inference.
  const string servable_name_;
  // Manager in charge of loading and unloading servables.
  std::unique_ptr<tensorflow::serving::Manager> manager_;
  // A scheduler for batching multiple request calls into single calls to
  // Session->Run().
  std::unique_ptr<tensorflow::serving::BasicBatchScheduler<Task>>
      batch_scheduler_;
};

// Take in the "service" instance (in this case representing an asynchronous
// server) and the completion queue "cq" used for asynchronous communication
// with the gRPC runtime.
CallData::CallData(InceptionServiceImpl* service_impl,
                   InceptionService::AsyncService* service,
                   ServerCompletionQueue* cq)
    : service_impl_(service_impl),
      service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
  // Invoke the serving logic right away.
  Proceed();
}

void CallData::Proceed() {
  if (status_ == CREATE) {
    // As part of the initial CREATE state, we *request* that the system
    // start processing Classify requests. In this request, "this" acts are
    // the tag uniquely identifying the request (so that different CallData
    // instances can serve different requests concurrently), in this case
    // the memory address of this CallData instance.
    service_->RequestClassify(&ctx_, &request_, &responder_, cq_, cq_, this);
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
  } else if (status_ == PROCESS) {
    // Spawn a new CallData instance to serve new clients while we process
    // the one for this CallData. The instance will deallocate itself as
    // part of its FINISH state.
    new CallData(service_impl_, service_, cq_);
    // Start processing.
    service_impl_->Classify(this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (CallData).
    delete this;
  }
}

void CallData::Finish(Status status) {
  status_ = FINISH;
  responder_.Finish(response_, status, this);
}

InceptionServiceImpl::InceptionServiceImpl(
    const string& servable_name,
    std::unique_ptr<tensorflow::serving::Manager> manager)
    : servable_name_(servable_name), manager_(std::move(manager)) {
  // Setup a batcher used to combine multiple requests (tasks) into a single
  // graph run for efficiency.
  // The batcher queues tasks until,
  //  (a) the next task would cause the batch to exceed the size target;
  //  (b) waiting for more tasks to be added would exceed the timeout.
  // at which point it processes the entire batch.
  //
  // Use the default batch-size, timeout and thread options.  In general
  // the numbers are extremely performance critical and should be tuned based
  // specific graph structure and usage.
  tensorflow::serving::BasicBatchScheduler<Task>::Options scheduler_options;
  scheduler_options.thread_pool_name = "inception_service_batch_threads";
  // Use a very large queue, to avoid rejecting requests. (Note: a production
  // server with load balancing may want to use the default, much smaller,
  // value.)
  scheduler_options.max_enqueued_batches = 1000;
  scheduler_options.max_batch_size = 10;
  TF_CHECK_OK(tensorflow::serving::BasicBatchScheduler<Task>::Create(
      scheduler_options,
      [this](std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
        this->DoClassifyInBatch(std::move(batch));
      },
      &batch_scheduler_));
}

// Creates a gRPC Status from a TensorFlow Status.
Status ToGRPCStatus(const tensorflow::Status& status) {
  return Status(static_cast<grpc::StatusCode>(status.code()),
                status.error_message());
}

void InceptionServiceImpl::Classify(CallData* calldata) {
  // Create and submit a task to the batch scheduler.
  std::unique_ptr<Task> task(new Task(calldata));
  tensorflow::Status status = batch_scheduler_->Schedule(&task);

  if (!status.ok()) {
    calldata->Finish(ToGRPCStatus(status));
    return;
  }
}

// Produces classifications for a batch of requests and associated responses.
void InceptionServiceImpl::DoClassifyInBatch(
    std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
  batch->WaitUntilClosed();
  if (batch->empty()) {
    return;
  }
  const int batch_size = batch->num_tasks();

  // Replies to each task with the given error status.
  auto complete_with_error = [&batch](StatusCode code, const string& msg) {
    Status status(code, msg);
    for (int i = 0; i < batch->num_tasks(); i++) {
      Task* task = batch->mutable_task(i);
      task->calldata->Finish(status);
    }
  };

  // Get a handle to the SessionBundle.  The handle ensures the Manager does
  // not reload this while it is in use.
  auto handle_request =
      tensorflow::serving::ServableRequest::Latest(servable_name_);
  tensorflow::serving::ServableHandle<tensorflow::serving::SessionBundle>
      bundle;
  const tensorflow::Status lookup_status =
      manager_->GetServableHandle(handle_request, &bundle);
  if (!lookup_status.ok()) {
    complete_with_error(StatusCode::INTERNAL,
                        lookup_status.error_message());
    return;
  }

  // Get the default signature of the graph.  Expected to be a
  // classification signature.
  tensorflow::serving::ClassificationSignature signature;
  const tensorflow::Status signature_status =
      GetClassificationSignature(bundle->meta_graph_def, &signature);
  if (!signature_status.ok()) {
    complete_with_error(StatusCode::INTERNAL,
                        signature_status.error_message());
    return;
  }

  // Transform protobuf input to inference input tensor.
  tensorflow::Tensor batched_input(tensorflow::DT_STRING, {batch_size});
  for (int i = 0; i < batch_size; ++i) {
    batched_input.vec<string>()(i) =
        batch->mutable_task(i)->calldata->request().jpeg_encoded();
  }

  // Run classification.
  tensorflow::Tensor batched_classes;
  tensorflow::Tensor batched_scores;
  const tensorflow::Status run_status =
      RunClassification(signature, batched_input, bundle->session.get(),
                        &batched_classes, &batched_scores);
  if (!run_status.ok()) {
    complete_with_error(StatusCode::INTERNAL, run_status.error_message());
    return;
  }

  // Transform inference output tensor to protobuf output.
  for (int i = 0; i < batch_size; ++i) {
    auto calldata = batch->mutable_task(i)->calldata;
    auto classes = calldata->mutable_response()->mutable_classes();
    auto scores = calldata->mutable_response()->mutable_scores();
    for (int j = 0; j < kNumTopClasses; j++) {
      *classes->Add() = batched_classes.matrix<string>()(i, j);
      scores->Add(batched_scores.matrix<float>()(i, j));
    }
    calldata->Finish(Status::OK);
  }
}

void HandleRpcs(InceptionServiceImpl* service_impl,
                InceptionService::AsyncService* service,
                ServerCompletionQueue* cq) {
  // Spawn a new CallData instance to serve new clients.
  new CallData(service_impl, service, cq);
  void* tag;  // uniquely identifies a request.
  bool ok = false;
  while (true) {
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    if (!cq->Next(&tag, &ok)) {
      break;  // server shutting down
    }
    if (!ok) {
      continue;  // irregular event
    }
    static_cast<CallData*>(tag)->Proceed();
  }
} */
void Classify(
    std::unique_ptr<tensorflow::serving::BasicBatchScheduler<Task>>* batch_scheduler,
    std::atomic_ulong& pred_counter, std::atomic_ulong& latency_sum_micros, InceptionRequest& request) {
    // if (request.image_data_size() != kImageDataSize) { 
    //         LOG(WARNING) << tensorflow::strings::StrCat(
    //     	          "expected image_data of size ", kImageDataSize,
    //     	          ", got ", request.image_data_size());
    //         return;
    // }

    // Create and submit a task to the batch scheduler.
    std::unique_ptr<Task> task(new Task(pred_counter, latency_sum_micros, request));
    tensorflow::Status status = (*batch_scheduler)->Schedule(&task);

    if (!status.ok()) {
        LOG(WARNING) << status.error_message();
    }
}

// Produces classifications for a batch of requests and associated responses.
void DoClassifyInBatch(
    const string servable_name,
    std::unique_ptr<tensorflow::serving::Manager>* manager,
     std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
    batch->WaitUntilClosed(); 
    if (batch->empty()) {
        return;
    }
    const int batch_size = batch->num_tasks();
    // Replies to each task with the given error status.
    auto complete_with_error = [&batch](const string& msg) {
        LOG(WARNING) << msg;
    };

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

    // Get the default signature of the graph.  Expected to be a
    // classification signature. 
    tensorflow::serving::ClassificationSignature signature;
    const tensorflow::Status signature_status =
        GetClassificationSignature(bundle->meta_graph_def, &signature);
    if (!signature_status.ok()) {
        complete_with_error(signature_status.error_message());
        return;
    }

    // Tensor input(tensorflow::DT_FLOAT, {kEvalBatchSize, kImageSize, kImageSize, kNumChannels});
    Tensor batched_input(tensorflow::DT_STRING, {kEvalBatchSize});
    // auto dst = input.flat_outer_dims<float>().data();
    for (int i = 0; i < batch_size; ++i) {
        batched_input.vec<string>()(i) = 
	    batch->mutable_task(i)->request.jpeg_encoded();    
        // std::copy_n(
        //     batch->mutable_task(i)->request.jpeg_encoded().begin(),
        //     kImageDataSize, dst);
        // dst += kImageDataSize;
    }

    // // pad to fill out batch size if needed
    for (int i = batch_size; i < kEvalBatchSize; ++i) {
        batched_input.vec<string>()(i) = 
	    batch->mutable_task(0)->request.jpeg_encoded();    
        // std::copy_n(
        //     batch->mutable_task(0)->request.jpeg_encoded().begin(),
        //     kImageDataSize, dst);
        // dst += kImageDataSize; 
    }

    // Run classification.
    std::cout << "Classifying\n";
    tensorflow::uint64 it_start = tensorflow::Env::Default()->NowMicros();
    tensorflow::Tensor scores;
    // const tensorflow::Status run_status =
    //     RunClassification(signature, input, bundle->session.get(),
    //                       nullptr /* classes */, &scores);
    const tensorflow::Status run_status =
        RunClassification(signature, batched_input, bundle->session.get(),
                          nullptr /* classes */, &scores);
    tensorflow::uint64 it_end = tensorflow::Env::Default()->NowMicros();
    tensorflow::uint64 latency = it_end - it_start;
    std::cout << "Done Classifying\n";
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
    }
    // Transform inference output tensor to protobuf output.
    // See mnist_model.py for details.
    const auto& scores_mat = scores.matrix<float>();
    if (batch_size > kEvalBatchSize) {
        LOG(INFO) <<
            tensorflow::strings::StrCat("BIG BATCH: ", batch_size);
    }
    for (int i = 0; i < batch_size; ++i) {
        batch->mutable_task(0)->pred_counter.fetch_add(1, std::memory_order::memory_order_relaxed);
        batch->mutable_task(0)->latency_sum_micros.fetch_add(latency, std::memory_order::memory_order_relaxed);
    }
}

void load_data(const string& path,
    std::unique_ptr<std::vector<InceptionRequest>>* data) {

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    std::unique_ptr<tensorflow::RandomAccessFile> imgFile;
    TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(path, &file));
    const size_t kBufferSizeBytes = 262144;
    tensorflow::io::InputBuffer in(file.get(), kBufferSizeBytes);
    string line;
    string imgLine;
    while (in.ReadLine(&line).ok()) {
        // std::vector<tensorflow::int32> parsed_line;
        // parsed_line.reserve(kImageDataSize + 1);
	InceptionRequest request;
    	TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(line, &imgFile));
        tensorflow::io::InputBuffer imgIn(imgFile.get(), kBufferSizeBytes);
	imgIn.ReadNBytes(kBufferSizeBytes, &imgLine).ok();
	request.set_jpeg_encoded(imgLine);
	std::cout << line << " " << imgLine.length() << "\n";
        (*data)->push_back(request);
	// request.set_jpeg_encoded(line);
        // bool parse_result = tensorflow::str_util::SplitAndParseAsInts(line, ',', &parsed_line);
        // std::vector<string> cols = str_util::Split(line, ',');
        // if (parse_result) {
        //     int label = parsed_line[0];
        //     InceptionRequest request;
        //     for (int i = 1; i < kImageDataSize + 1; ++i) {
        //         // request.set_jpeg_encoded(parsed_line[i] + "");
        //         request.add_image_data(parsed_line[i] * 1.0 / 255.0);
        //     }
	//     // request.set_jpeg_encoded(request.get_jpeg_encoded() + line);
        //     (*data)->push_back(request);
        // } else {
        //     LOG(INFO) << "parsing error";
        // }
    }
    LOG(INFO) << tensorflow::strings::StrCat("Loaded ", (*data)->size(), " test inputs");
}

// Runs InceptionService server until shutdown.
/*
void RunServer(const int port, const string& servable_name,
               std::unique_ptr<tensorflow::serving::Manager> manager) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);

  InceptionService::AsyncService service;
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<ServerCompletionQueue> cq = builder.AddCompletionQueue();
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running...";

  InceptionServiceImpl service_impl(servable_name, std::move(manager));
  HandleRpcs(&service_impl, &service, cq.get());
} */

// }  // namespace

int main(int argc, char** argv) {
  // Parse command-line options.
  string inception_data_path; 
  tensorflow::int32 num_requests;
  tensorflow::int32 num_batch_threads;
  const bool parse_result =
      tensorflow::ParseFlags(&argc, argv, {
        tensorflow::Flag("inception_data_file", &inception_data_path),
        tensorflow::Flag("num_requests", &num_requests),
        tensorflow::Flag("num_batch_threads", &num_batch_threads),
	tensorflow::Flag("batch_size", &kEvalBatchSize),
        });
  if (!parse_result) {
    LOG(FATAL) << "Error parsing command line flags.";
  }

  // tensorflow::int32 port = 0;
  // const bool parse_result =
  //     tensorflow::ParseFlags(&argc, argv, {tensorflow::Flag("port", &port)});
  // if (!parse_result) {
  //   LOG(FATAL) << "Error parsing command line flags.";
  // }

  // if (argc != 2) {
  //   LOG(ERROR) << "Usage: inception_inference --port=9000 /path/to/exports";
  //   return -1;
  // }
  // const string export_base_path(argv[1]);
  const string servable_path(argv[1]);
  std::unique_ptr<std::vector<InceptionRequest>> input_data(new std::vector<InceptionRequest>);
  load_data(inception_data_path, &input_data);
  std::atomic_ulong pred_counter;
  pred_counter.store(0);

  std::atomic_ulong latency_sum_micros;
  latency_sum_micros.store(0);

  tensorflow::port::InitMain(argv[0], &argc, &argv);


  std::unique_ptr<tensorflow::serving::Manager> manager;
  tensorflow::Status status = tensorflow::serving::simple_servers::
      CreateSingleTFModelManagerFromBasePath(servable_path, &manager);
      // CreateSingleTFModelManagerFromBasePath(export_base_path, &manager);

  TF_CHECK_OK(status) << "Error creating manager";

  // Wait until at least one model is loaded.
  std::vector<tensorflow::serving::ServableId> ready_ids;
  // TODO(b/25545573): Create a more streamlined startup mechanism than polling.
  do {
    LOG(INFO) << "Waiting for models to be loaded...";
    tensorflow::Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /*1 sec*/);
    ready_ids = manager->ListAvailableServableIds();
  } while (ready_ids.empty());
  std::cout << "model loaded\n";

  // A scheduler for batching multiple request calls into single calls to
  // Session->Run().
  std::unique_ptr<tensorflow::serving::BasicBatchScheduler<Task>>
      batch_scheduler;

  tensorflow::serving::BasicBatchScheduler<Task>::Options scheduler_options;
  scheduler_options.thread_pool_name = "inception_service_batch_threads";
  scheduler_options.num_batch_threads = num_batch_threads;
  scheduler_options.batch_timeout_micros = 0;
  scheduler_options.max_batch_size = kEvalBatchSize;
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
  std::cout << "input data size: " << input_data->size() << "\n";

  tensorflow::uint64 start_time = tensorflow::Env::Default()->NowMicros();
  for (int i = 0; i < num_requests; ++i) {
      Classify(&batch_scheduler, pred_counter, latency_sum_micros, (*input_data)[mnist_dist(rng)]);
  }
  std::cout << "put all on queue\n";

  while (batch_scheduler->NumEnqueuedTasks() > 0) {
    tensorflow::Env::Default()->SleepForMicroseconds(10 * 1000 /*0.5 sec*/);
  }

  tensorflow::uint64 end_time = tensorflow::Env::Default()->NowMicros();
  tensorflow::uint64 processed_reqs = pred_counter.load();
  tensorflow::uint64 total_batch_latency = latency_sum_micros.load();
  double mean_latency = (double) total_batch_latency / (double) processed_reqs;
  tensorflow::uint64 total_micros = (end_time - start_time);
  double total_seconds = total_micros / (1000.0 * 1000.0);

  double throughput = ((double) processed_reqs) / total_seconds;
  LOG(INFO) << tensorflow::strings::StrCat("Processed ", processed_reqs,
      " predictions in ", total_micros, " micros. Throughput: ",
      throughput, " Mean Latency: ", mean_latency);

  // Run the service.
  // RunServer(port, ready_ids[0].name, std::move(manager));

  return 0;
}

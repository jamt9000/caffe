// Copyright Yangqing Jia 2013

#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>
#include <boost/asio.hpp>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/distributed_solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using boost::asio::ip::tcp;


namespace caffe {

template <typename Dtype>
void DistributedSolverParamServer<Dtype>::Solve(const char* resume_file) {
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << this->net_->name();

  this->iter_ = 0;
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Solver<Dtype>::Restore(resume_file);
  }
  next_snapshot_ = this->iter_ + this->param_.snapshot();

  // the main loop.
  LOG(INFO) << "Waiting for incoming updates...";
  while (this->iter_ < this->param_.max_iter()) {
    ReceiveAndSend();
    // Check if we need to do snapshot
    if (this->param_.snapshot() && this->iter_ > next_snapshot_) {
      Solver<Dtype>::Snapshot();
      next_snapshot_ += this->param_.snapshot();
    }
    // TODO: test
  }
  LOG(INFO) << "Optimization Done.";
}


// Receive and send: what this function does is to get the accumulated gradient
// values from the client, stores it to the diff field of the network, and then
// updates the network. It then sends back the updated network value to the
// client.
template <typename Dtype>
void DistributedSolverParamServer<Dtype>::ReceiveAndSend() {
  bool send_only;
  int incoming_iter;

  boost::asio::io_service io_s;
  tcp::acceptor data_acceptor(
      io_s, tcp::endpoint(tcp::v4(), atoi(this->param_.tcp_port().c_str())));
  tcp::iostream data_stream;
  data_acceptor.accept(*(data_stream.rdbuf()));
  LOG(INFO) << "Incoming connection.";
  data_stream.read(reinterpret_cast<char*>(&send_only), sizeof(send_only));
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  if (!send_only) {
    // Receive data
    LOG(INFO) << "Receiving data.";
    data_stream.read(reinterpret_cast<char*>(&incoming_iter),
        sizeof(incoming_iter));
    int total_received = 0;
    LOG(INFO) << "Incoming iterations: " << incoming_iter;
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype* param_diff = net_params[param_id]->mutable_cpu_diff();
      int count = net_params[param_id]->count();
      data_stream.read(reinterpret_cast<char*>(param_diff),
          count * sizeof(Dtype));
      total_received += count;
    }
    LOG(INFO) << "Received " << total_received << " variables.";
    // Check error: if there are any error in the receiving phase, we will not
    // trust the passed in update.
    if (data_stream.error()) {
      LOG(ERROR) << "Error in receiving. Error code: " << data_stream.error().message();
    } else {
      // If the read is successful, update the network.
      this->iter_ += incoming_iter;
      this->net_->Update();
    }
  } else {
    LOG(INFO) << "No incoming updates. Will simply send data.";
  }
  // Send data
  LOG(INFO) << "Sending data";
  data_stream.write(reinterpret_cast<char*>(&(this->iter_)), sizeof(this->iter_));
  LOG(INFO) << "Current iteration: " << this->iter_;
  int total_sent = 0;
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    const Dtype* param_data = net_params[param_id]->cpu_data();
    int count = net_params[param_id]->count();
    data_stream.write(reinterpret_cast<const char*>(param_data),
        sizeof(Dtype) * count);
    total_sent += count;
  }
  LOG(INFO) << "Sent " << total_sent << " variables.";
  data_stream.flush();
}


template <typename Dtype>
void DistributedSolverParamClient<Dtype>::Solve(const char* resume_file) {
  // Although we have resume_file, the client never does the actual resuming.
  // Instead, it will simply request the weights from the server.
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << this->net_->name();
  PreSolve();

  // Send and receive once to get the current iteration and the parameters
  LOG(INFO) << "Obtaining initial parameters.";
  SendAndReceive(true);
  LOG(INFO) << "Initial communication finished.";

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  next_display_ = this->iter_ + this->param_.display();
  while (this->iter_++ < this->param_.max_iter()) {
    Dtype loss = this->net_->ForwardBackward(bottom_vec);
    ComputeUpdateValue();
    this->net_->Update();

    if (this->param_.display() && this->iter_ > next_display_) {
      LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss;
      next_display_ = this->iter_ + this->param_.display();
    }
  }
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void DistributedSolverParamClient<Dtype>::SendAndReceive(bool receive_only) {
  tcp::iostream data_stream(this->param_.tcp_server(), this->param_.tcp_port());
  if (!data_stream) {
    LOG(FATAL) << "Unable to connect. Error code: " << data_stream.error().message();
  }
  data_stream.write(reinterpret_cast<char*>(&receive_only), sizeof(receive_only));
  if (!receive_only) {
    LOG(INFO) << "Sending local changes.";
    int local_iters = this->param_.communication_interval();
    data_stream.write(reinterpret_cast<char*>(&local_iters),
        sizeof(local_iters));
    int total_sent = 0;
    // TODO: send the accumulated gradient stored at history_, and set it to
    // zero for future accumulation
    for (int param_id = 0; param_id < this->history_.size(); ++param_id) {
      Dtype* accum_history_data = this->history_[param_id]->mutable_cpu_diff();
      int count = this->history_[param_id]->count();
      data_stream.write(reinterpret_cast<char*>(accum_history_data),
          sizeof(Dtype) * count);
      memset(accum_history_data, 0, sizeof(Dtype) * count);
      total_sent += count;
    }
    LOG(INFO) << "Sent " << total_sent << " variables.";
    CHECK(!data_stream.error()) << "Error in sending. Error code: "
        << data_stream.error().message();
  }// else {
  //  LOG(INFO) << "Not sending local changes. Receive only.";
  //}
  data_stream.flush();
  // Receive parameters
  LOG(INFO) << "Receiving parameters.";
  data_stream.read(reinterpret_cast<char*>(&(this->iter_)),
      sizeof(this->iter_));
  LOG(INFO) << "New iteration: " << this->iter_;
  int total_received = 0;
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    Dtype* param_data = net_params[param_id]->mutable_cpu_data();
    int count = net_params[param_id]->count();
    data_stream.read(reinterpret_cast<char*>(param_data),
        sizeof(Dtype) * count);
    total_received += count;
    // Also, let's set the param_diff to be zero so that this update does not
    // change the parameter value, since it has already been updated.
    memset(net_params[param_id]->mutable_cpu_diff(), 0,
        net_params[param_id]->count() * sizeof(Dtype));
  }
  CHECK(!data_stream.error()) << "Error in communication. Error code: "
      << data_stream.error().message();
  LOG(INFO) << "Received " << total_received << " variables.";
  // Set the next send iter.
  next_send_iter_ = this->iter_ + this->param_.communication_interval();
}


template <typename Dtype>
void DistributedSolverParamClient<Dtype>::ComputeUpdateValue() {
  // First, carry out the normal update
  SGDSolver<Dtype>::ComputeUpdateValue();
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  // Accumulate the gradient history
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      caffe_axpy(net_params[param_id]->count(), Dtype(1.),
          net_params[param_id]->cpu_diff(),
          this->history_[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      caffe_gpu_axpy(net_params[param_id]->count(), Dtype(1.),
          net_params[param_id]->gpu_diff(),
          this->history_[param_id]->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  // See if we need to do communication.
  if (this->iter_ >= next_send_iter_) {
    SendAndReceive();
  }
}


INSTANTIATE_CLASS(DistributedSolverParamServer);
INSTANTIATE_CLASS(DistributedSolverParamClient);


}  // namespace caffe

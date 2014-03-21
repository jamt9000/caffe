// Copyright 2013 Yangqing Jia
// This implements the distributed solver in two classes: the parameter server
// that holds the parameters and the actual solver that does computation.

#include <string>
#include <vector>
#include <mpi.h>
#include <glog/logging.h>

#include "caffe/distributed_solver.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
// ===================================
// MPI initialization and finalization
// ===================================

/**
 * @brief Initialize the MPI environment, allowing the CUDA device to be selected before (if necessary)
 *
 * @param[in, out]        argc        The number of command-line arguments
 * @param[in, out]        argv        The list of command-line arguments
 * @param[out]                rank        The global rank of the current MPI process
 * @param[out]                size        The total number of MPI processes available
 */
void InitializeMPI(int* argc, char*** argv, int* rank, int* size) {
  Util<>::checkComputeMode();
  if (Caffe::mode() == Caffe::GPU) {
    // Setting the device here will have an effect only for the CUDA-aware MPI version
    SetDeviceBeforeInit();
  }

  MPI_Init(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, size);
}

/**
 * @brief Close (finalize) the MPI environment and deallocate buffers
 */
void FinalizeMPI() {
  MPI_Finalize();
}

void SetDeviceBeforeInit() {
  char * localRankStr = NULL;
  int rank = 0, devCount = 0;
  // We extract the local rank initialization using an environment variable
  if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL) {
    rank = atoi(localRankStr);
  }
  CUDA_CHECK(cudaGetDeviceCount(&devCount));
  CUDA_CHECK(cudaSetDevice(rank % devCount));
}

template<typename Dtype>
MPI_Datatype Util<Dtype>::GetMpiFloatingPointType() {
  if (sizeof(Dtype) == sizeof(float)) {
    return MPI_FLOAT;
  } else if (sizeof(Dtype) == sizeof(double)) {
    return MPI_DOUBLE;
  } else {
    LOG(FATAL)<< "Unknown float point data type.";
  }
}

template<typename Dtype>
Dtype* Util<Dtype>::GetMutableData(const shared_ptr<Blob<Dtype> > blob_ptr) {
  Dtype* ptr;
  switch (Caffe::mode()) {
    case Caffe::GPU:
      // Here it is not necessary to use host buffers and intermediate memory transfers
      // because we are using CUDA aware MPI such as MVAPICH2 or Open MPI
      ptr = blob_ptr->mutable_gpu_diff();
      break;
    case Caffe::CPU:
      ptr = blob_ptr->mutable_cpu_diff();
      break;
    default:
      LOG(FATAL) << "Unknown float point data type.";
  }
}

// Used to fail fast, fail early
template<typename Dtype>
void Util<Dtype>::checkComputeMode() {
  switch (Caffe::mode()) {
    case Caffe::GPU:
    case Caffe::CPU:
      break;
    default:
      LOG(FATAL) << "Unknown float point data type.";
  }
}

template<typename Dtype>
SolverServer<Dtype>::SolverServer(const SolverParameter& param,
                                  const vector<int>& client_ranks)
    : Solver<Dtype>(param),
      mpi_real_type_(Util<Dtype>::GetMpiFloatingPointType()),
      client_ranks_(client_ranks),
      mpi_tag_(0),
      mpi_comm_(MPI_COMM_WORLD) {
  Util<Dtype>::checkComputeMode();
  if (client_ranks_.size() <= 0) {
    LOG(FATAL) << "No solver client to start training";
  }
}

template<typename Dtype>
SolverServer<Dtype>::~SolverServer() {
}

template<typename Dtype>
void SolverServer<Dtype>::Solve(const char* resume_file) {
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO)<< "Solving " << this->net_->name();

  this->iter_ = 0;
  if (resume_file) {
    LOG(INFO)<< "Restoring previous solver status from " << resume_file;
    Solver<Dtype>::Restore(resume_file);
  }
  next_snapshot_ = this->iter_ + this->param_.snapshot();

  // the main loop.
  LOG(INFO)<< "Waiting for incoming updates...";
  while (this->iter_ < this->param_.max_iter()) {
    ReceiveAndSend();
    // Check if we need to do snapshot
    if (this->param_.snapshot() && this->iter_ > next_snapshot_) {
      Solver<Dtype>::Snapshot();
      next_snapshot_ += this->param_.snapshot();
    }
    // TODO: test
  }
  LOG(INFO)<< "Optimization Done.";
}

// The distributed solver does not do solving itself.
template<typename Dtype>
void SolverServer<Dtype>::ComputeUpdateValue() {
}

// The distributed solver has nothing to snapshot.
template<typename Dtype>
void SolverServer<Dtype>::SnapshotSolverState(SolverState* state) {
}

template<typename Dtype>
void SolverServer<Dtype>::RestoreSolverState(const SolverState& state) {
}

// Receive and send: what this function does is to get the accumulated gradient
// values from the client, stores it to the diff field of the network, and then
// updates the network. It then sends back the updated network value to the
// client.
template<typename Dtype>
void SolverServer<Dtype>::ReceiveAndSend() {
  int send_only;
  int incoming_iter;
  int client_rank;
  MPI_Status status;
  LOG(INFO)<< "Incoming connection.";

  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  for (size_t num = 0; num < client_ranks_.size(); num++) {
    client_rank = client_ranks_[num];
    MPI_Recv(reinterpret_cast<void*>(send_only), sizeof(send_only), MPI_INT,
             client_rank, mpi_tag_, mpi_comm_, &status);
    MPI_CHECK(&status, 1, MPI_INT);
    if (!send_only) {
      // Receive data
      LOG(INFO)<< "Receiving data from client " << client_rank << ".";
      MPI_Recv(reinterpret_cast<void*>(incoming_iter), sizeof(incoming_iter), MPI_INT,
          client_rank, mpi_tag_, mpi_comm_, &status);
      MPI_CHECK(&status, 1, MPI_INT);
      int total_received = 0;
      LOG(INFO) << "Incoming iterations: " << incoming_iter << " from client " << client_rank << ".";
      for (int param_id = 0; param_id < net_params.size(); ++param_id) {
        Dtype* param_diff = Util<Dtype>::GetMutableData(net_params[param_id]);
        int count = net_params[param_id]->count();
        // TODO: MPI_IRecv asynchronous receive
        MPI_Recv(reinterpret_cast<void*>(param_diff), count * sizeof(Dtype),
            mpi_real_type_, client_rank, mpi_tag_, mpi_comm_, &status);
        MPI_CHECK(&status, count, mpi_real_type_);
        total_received += count;
      }
      LOG(INFO) << "Received " << total_received << " variables.";
      // Check error: if there are any error in the receiving phase, we will not
      // trust the passed in update.
      this->iter_ += incoming_iter;
      this->net_->Update();
    }
    else {
      LOG(INFO) << "No incoming updates from client " << client_rank << ". Will simply send data.";
    }
    // Send data
    LOG(INFO)<< "Sending data to client " << client_rank << ".";
    MPI_Send(reinterpret_cast<void*>(this->iter_), sizeof(this->iter_),
    MPI_INT,
             client_rank, mpi_tag_, mpi_comm_);
    LOG(INFO)<< "Current iteration: " << this->iter_;
    int total_sent = 0;
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype* param_data = Util<Dtype>::GetMutableData(net_params[param_id]);
      int count = net_params[param_id]->count();
      MPI_Send(reinterpret_cast<void*>(param_data), count * sizeof(Dtype),
               mpi_real_type_, client_rank, mpi_tag_, mpi_comm_);
      total_sent += count;
    }
    LOG(INFO)<<"Sent " << total_sent << " variables to client " << client_rank << ".";
  }  // for (size num = 0; num < client_ranks_.size(); num++) {
}

template<typename Dtype>
SGDSolverClient<Dtype>::SGDSolverClient(const SolverParameter& param,
                                        const int server_rank)
    : SGDSolver<Dtype>(param),
      mpi_real_type_(Util<Dtype>::GetMpiFloatingPointType()),
      server_rank_(server_rank),
      mpi_tag_(0),
      mpi_comm_(MPI_COMM_WORLD) {
  Util<Dtype>::checkComputeMode();
  CHECK_GT(param.communication_interval(), 0);
  next_send_iter_ = param.communication_interval();
}

template<typename Dtype>
void SGDSolverClient<Dtype>::Solve(const char* resume_file) {
// Although we have resume_file, the client never does the actual resuming.
// Instead, it will simply request the weights from the server.
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO)<< "Solving " << this->net_->name();
  PreSolve();

// Send and receive once to get the current iteration and the parameters
  LOG(INFO)<< "Obtaining initial parameters.";
  SendAndReceive(true);
  LOG(INFO)<< "Initial communication finished.";

// For a network that is trained by the solver, no bottom or top vecs
// should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  next_display_ = this->iter_ + this->param_.display();
  while (this->iter_++ < this->param_.max_iter()) {
    Dtype loss = this->net_->ForwardBackward(bottom_vec);
    ComputeUpdateValue();
    this->net_->Update();

    if (this->param_.display() && this->iter_ > next_display_) {
      LOG(INFO)<< "Iteration " << this->iter_ << ", loss = " << loss;
      next_display_ = this->iter_ + this->param_.display();
    }
  }
  LOG(INFO)<< "Optimization Done.";
}

template<typename Dtype>
void SGDSolverClient<Dtype>::PreSolve() {
  SGDSolver<Dtype>::PreSolve();
}

template<typename Dtype>
void SGDSolverClient<Dtype>::SendAndReceive(const bool receive_only) {
  int receive_only_int = receive_only ? 1 : 0;
  MPI_Send(reinterpret_cast<void*>(&receive_only_int), sizeof(receive_only_int),
  MPI_INT,
           server_rank_, mpi_tag_, mpi_comm_);
  if (!receive_only) {
    LOG(INFO)<< "Sending local changes.";
    // Not using this->param_.communication_interval()
    // since this->iter_ - last_sending_iter_
    // may not be exactly equal to the former
    int local_iters = this->iter_ - last_sending_iter_;
    MPI_Send(reinterpret_cast<void*>(&local_iters), sizeof(local_iters),
        MPI_INT, server_rank_, mpi_tag_, mpi_comm_);
    int total_sent = 0;
    // TODO: send the accumulated gradient stored at history_, and set it to
    // zero for future accumulation
    for (int param_id = 0; param_id < this->history_.size(); ++param_id) {
      Dtype* accum_history_data = Util<Dtype>::GetMutableData(this->history_[param_id]);
      int count = this->history_[param_id]->count();
      MPI_Send(reinterpret_cast<void*>(accum_history_data), count * sizeof(Dtype),
          mpi_real_type_, server_rank_, mpi_tag_, mpi_comm_);
      memset(accum_history_data, 0, sizeof(Dtype) * count);
      total_sent += count;
    }
    LOG(INFO) << "Sent " << total_sent << " variables.";
  }
  // Receive parameters
  LOG(INFO)<< "Receiving parameters.";
  MPI_Status status;
  MPI_Recv(reinterpret_cast<void*>(&(this->iter_)), sizeof(this->iter_),
  MPI_INT,
           server_rank_, mpi_tag_, mpi_comm_, &status);
  MPI_CHECK(&status, 1, MPI_INT);
  last_sending_iter_ = this->iter_;
  LOG(INFO)<< "New iteration: " << this->iter_;
  int total_received = 0;
  vector<shared_ptr<Blob<Dtype> > > &net_params = this->net_->params();
  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
    Dtype* param_data = Util<Dtype>::GetMutableData(net_params[param_id]);
    int count = net_params[param_id]->count();
    MPI_Recv(reinterpret_cast<void*>(param_data), count * sizeof(Dtype),
             mpi_real_type_, server_rank_, mpi_tag_, mpi_comm_, &status);
    MPI_CHECK(&status, count, mpi_real_type_);
    total_received += count;
    // Also, let's set the param_diff to be zero so that this update does not
    // change the parameter value, since it has already been updated.
    memset(net_params[param_id]->mutable_cpu_diff(), 0,
           net_params[param_id]->count() * sizeof(Dtype));
  }
  LOG(INFO)<< "Received " << total_received << " variables from server " << server_rank_ << ".";
  // Set the next send iter.
  next_send_iter_ = this->iter_ + this->param_.communication_interval();
}

template<typename Dtype>
void SGDSolverClient<Dtype>::ComputeUpdateValue() {
  // First, carry out the normal update
  SGDSolver<Dtype>::ComputeUpdateValue();
  vector<shared_ptr<Blob<Dtype> > > &net_params = this->net_->params();
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
      LOG(FATAL)<< "Unknown caffe mode.";
    }
    // See if we need to do communication.
  if (this->iter_ >= next_send_iter_) {
    SendAndReceive();
  }
}

INSTANTIATE_CLASS(Util);
INSTANTIATE_CLASS(SolverServer);
INSTANTIATE_CLASS(SGDSolverClient);

}  // namespace caffe

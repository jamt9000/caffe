// Copyright 2013 Yangqing Jia
// This implements the distributed solver in two classes: the parameter server
// that holds the parameters and the actual solver that does computation.

// Renaissance 2014 @kloudkl

#ifndef CAFFE_DISTRIBUTED_SOLVER_HPP_
#define CAFFE_DISTRIBUTED_SOLVER_HPP_

#include <cstdlib> // for NULL
#include <vector>
#include <mpi.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

#define USE_MVAPICH2

// The following MPI related macros and utility functions are borrowed from
// https://github.com/parallel-forall/code-samples/blob/master/posts/cuda-aware-mpi-example/
// authored by @jirikraus
// TODO: Check if Caffe can use, reproduce, disclose, or distribute them
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
/**
 * This is the environment variable which allows the reading of the local rank of the current MPI
 * process before the MPI environment gets initialized with MPI_Init(). This is necessary when running
 * the CUDA-aware MPI version of the Jacobi solver, which needs this information in order to be able to
 * set the CUDA device for the MPI process before MPI environment initialization. If you are using MVAPICH2,
 * set this constant to "MV2_COMM_WORLD_LOCAL_RANK"; for Open MPI, use "OMPI_COMM_WORLD_LOCAL_RANK".
 */
#if defined USE_MVAPICH2
#define ENV_LOCAL_RANK                "MV2_COMM_WORLD_LOCAL_RANK"
#elif defined USE_OPEN_MPI
#define ENV_LOCAL_RANK                "OMPI_COMM_WORLD_LOCAL_RANK"
#endif

/**
 * This is the global rank of the root (master) process
 */
#define MPI_MASTER_RANK                0
#define STATUS_ERR                     -1
#define MPI_CHECK(status, expectedElems, mpi_real_type) \
  do { \
    int recvElems; \
    MPI_Get_count(status, mpi_real_type, &recvElems); \
    if (recvElems != expectedElems) { \
      LOG(ERROR) << "MPI transfer returned " << recvElems \
        << " elements, but " << expectedElems << " were expected. " \
          "Terminating...\n"; \
      exit (STATUS_ERR); \
    } \
  } while(0)

namespace caffe {
// =============
// Host routines
// =============
void InitializeMPI(int* argc, char*** argv, int* rank, int* size);
void FinalizeMPI();
void SetDeviceBeforeInit();

template<typename Dtype = float>
class Util {
 public:
  Util();
  static MPI_Datatype GetMpiFloatingPointType();
  static Dtype* GetMutableData(const shared_ptr<Blob<Dtype> >);
  static void checkComputeMode();
};

template<typename Dtype>
class SolverServer : public Solver<Dtype> {
 public:
  SolverServer(const SolverParameter& param, const vector<int>& client_ranks);
  virtual ~SolverServer();
  virtual void Solve(const char* resume_file = NULL);
 protected:
  virtual void ComputeUpdateValue();
  virtual void SnapshotSolverState(SolverState* state);
  virtual void RestoreSolverState(const SolverState& state);
  // The function that implements the actual communication.
  void ReceiveAndSend();
 protected:
  size_t next_snapshot_;
  MPI_Datatype mpi_real_type_;
  vector<int> client_ranks_;
  int mpi_tag_;
  MPI_Comm mpi_comm_;
};

template<typename Dtype>
class SGDSolverClient : public SGDSolver<Dtype> {
 public:
  SGDSolverClient(const SolverParameter& param, const int server_rank);
  virtual void Solve(const char* resume_file = NULL);
 protected:
  virtual void PreSolve();
  virtual void ComputeUpdateValue();
  void SendAndReceive(bool just_receive = false);
 protected:
  size_t next_send_iter_;
  size_t next_display_;
  MPI_Datatype mpi_real_type_;
  int server_rank_;
  int mpi_tag_;
  MPI_Comm mpi_comm_;
  size_t last_sending_iter_;
};

}  // namespace caffe

#endif /* DISTRIBUTED_SOLVER_HPP_ */

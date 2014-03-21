// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to train a network using
//     multiple CPUs/GPUs on a single server or in a cluster
// Usage:
//    distributed_training solver_proto_file [resume_point_file]

#include <cuda_runtime.h>

#include <algorithm> // for min
#include <string>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp> // for is_any_of

#include "caffe/caffe.hpp"
#include "caffe/distributed_solver.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR)<< "\nUsage: distributed_training solver_proto_files [resume_point_file]\n"
    "         set multiple solver_proto_files for processes using colon(:)\n"
    "         eg. running: mpirun -np 8 distributed_training sp1:sp2:sp3\n"
    "         the 1st, 2nd & 3rd processes will use sp1, sp2, sp3 respectively\n"
    "         all the rest processes will use sp3\n";
    return 0;
  }

  std::string protos(argv[1]);
  std::vector<std::string> solver_proto_files;
  std::string seperator = ":";
  boost::algorithm::split(solver_proto_files, protos,
                          boost::algorithm::is_any_of(seperator));

  int rank;
  int size;
  InitializeMPI(&argc, &argv, &rank, &size);
  SolverParameter solver_param;
  // This does not work on a single machine using multiple processes
  // because leveldb forbids them accessing at the same time
  // Have to use different leveldb copies and therefore solver_proto_files
//  ReadProtoFromTextFile(argv[1], &solver_param);
  int use_which_solver_proto_file = std::min<int>(
      rank, solver_proto_files.size() - 1);

  LOG(INFO)<< "Starting Optimization";
  // GPU single precision performance is way higher than double precision performance
  // Refer to http://en.wikipedia.org/wiki/GeForce_700_Series
  shared_ptr<Solver<float> > solver;
  if (rank == MPI_MASTER_RANK) {
    // Simulate a cluster on a machine with only one GPU
    Caffe::set_mode(Caffe::CPU);
    vector<int> client_ranks;
    for (int s = 0; s < size; ++s) {
      if (s != MPI_MASTER_RANK) {
        client_ranks.push_back(s);
      }
    }
    printf("ReadProtoFromTextFile rank %d\n", rank);
    ReadProtoFromTextFile(solver_proto_files[use_which_solver_proto_file],
                          &solver_param);
    solver.reset(new SolverServer<float>(solver_param, client_ranks));
  } else {
    Caffe::set_mode(Caffe::CPU);
//    Caffe::set_mode(Caffe::GPU);
    printf("ReadProtoFromTextFile rank %d\n", rank);
    ReadProtoFromTextFile(solver_proto_files[use_which_solver_proto_file],
                          &solver_param);
    solver.reset(new SGDSolverClient<float>(solver_param, MPI_MASTER_RANK));
  }

  if (argc > 2) {
    LOG(INFO)<< "Resuming from " << argv[2];
    solver->Solve(argv[2]);
  } else {
    printf("Solve rank %d\n", rank);
    if (rank != -1) {
      solver->Solve();
    }
  }
  FinalizeMPI();
  LOG(INFO)<< "Optimization Done.";

  return 0;
}

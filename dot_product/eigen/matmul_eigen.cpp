#include <iostream>
#include <Eigen/Dense>
#include <chrono>

double benchmark_matmul_cpp(int M, int N, int K) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(M, K);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(K, N);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(M, N);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for(int i=0; i<2; ++i){
    C.noalias() += A * B; // Eigen provides optimized matrix multiplication
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;

  return ((2.0 * M * N * K) / (diff.count() / 2)) / 1e9;
}

int main() {
  int M = 1024, N = 1024, K = 1024;
  double gflops = benchmark_matmul_cpp(M, N, K);
  std::cout << gflops << " GFLOP/s" << std::endl;

  return 0;
}
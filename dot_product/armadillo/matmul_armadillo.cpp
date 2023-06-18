#include <iostream>
#include <armadillo>
#include <chrono>

// Function to benchmark matrix multiplication
void benchmark_matmul(int M, int N, int K) {
    // Create random matrices A and B
    arma::mat A = arma::randu<arma::mat>(M, K);
    arma::mat B = arma::randu<arma::mat>(K, N);

    // Perform multiplication and measure time
    auto start = std::chrono::high_resolution_clock::now();
    arma::mat C = A * B;
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time in seconds
    std::chrono::duration<double> elapsed = end - start;

    // Calculate and print GFLOPS
    double gflops = 2.0 * M * N * K / (elapsed.count() * 1e9);
    std::cout << gflops << " GFLOP/s" << std::endl;
}

int main() {
    int M = 1024; // rows of A and C
    int N = 1024; // columns of B and C
    int K = 1024; // columns of A and rows of B

    // Benchmark matrix multiplication
    benchmark_matmul(M, N, K);

    return 0;
}

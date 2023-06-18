#include <iostream>
#include <armadillo>

void benchmark(int M, int N) {
    arma::fmat A = arma::randu<arma::fmat>(M, N);
    arma::fmat B = arma::randu<arma::fmat>(M, N);

    auto start = std::chrono::high_resolution_clock::now();
    arma::fmat C = A % B;
    auto stop = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double elapsed = static_cast<double>(duration.count()) * 1e-6; // Convert microseconds to seconds
    double gflops = static_cast<double>(M * N) / (elapsed * 1e9); // Only one operation per element pair for element-wise multiplication
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;
}

int main() {
    int M = 1024;
    int N = 1024;

    benchmark(M, N);
    
    return 0;
}
#include <iostream>
#include <immintrin.h>
#include <vector>
#include <chrono>

void multiply(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(A + i*K + k);
                __m256 b_vec = _mm256_loadu_ps(B + j*K + k);
                __m256 mult_vec = _mm256_mul_ps(a_vec, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, mult_vec);
            }
            _mm256_storeu_ps(C + i*N + j, sum_vec);
        }
    }
}

void benchmark_matmul(int M, int N, int K) {
    float *A = new float[M*K];
    float *B = new float[K*N];
    float *C = new float[M*N];

    std::fill(A, A + M*K, 0.5f);
    std::fill(B, B + K*N, 0.5f);

    auto start = std::chrono::high_resolution_clock::now();

    multiply(A, B, C, M, N, K);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    double gflops = 2.0 * M * N * K / (elapsed.count() * 1e9);
    std::cout << gflops << " GFLOP/s" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    benchmark_matmul(M, N, K);

    return 0;
}
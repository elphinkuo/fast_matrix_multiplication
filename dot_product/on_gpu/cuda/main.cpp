// Filename: main.cpp

#include <iostream>
#include <chrono>
#include <random>

extern void matrixMul(float *A, float *B, float *C);

const int N = 1024;

int main() {
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    // Random number generation
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();

    matrixMul(A, B, C);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << " s\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

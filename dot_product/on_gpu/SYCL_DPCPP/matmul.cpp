#include <CL/sycl.hpp>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace cl::sycl;

class MatMulKernel;

int main() {
    // Define matrix size
    const int N = 1024;
    std::vector<float> A(N * N), B(N * N), C(N * N);

    // Initialize matrix A with a sine-wave pattern
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = std::sin(i * 0.1) * std::sin(j * 0.1);
        }
    }

    // Initialize matrix B with a cosine-wave pattern
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i * N + j] = std::cos(i * 0.1) * std::cos(j * 0.1);
        }
    }

    // Create a SYCL queue with default selector and enable profiling
    queue q(default_selector{}, property::queue::enable_profiling{});

    // Create buffers
    buffer<float, 2> bufA(A.data(), range<2>(N, N));
    buffer<float, 2> bufB(B.data(), range<2>(N, N));
    buffer<float, 2> bufC(C.data(), range<2>(N, N));

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Submit command group to queue
    event e = q.submit([&](handler& h) {
        // Accessors
        auto accA = bufA.get_access<access::mode::read>(h);
        auto accB = bufB.get_access<access::mode::read>(h);
        auto accC = bufC.get_access<access::mode::write>(h);

        // Matrix multiplication
        h.parallel_for<class MatMulKernel>(range<2>(N, N), [=](id<2> idx) {
            int x = idx[0], y = idx[1];
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += accA[x][i] * accB[i][y];
            }
            accC[x][y] = sum;
        });
    });

    // Wait for event to complete
    e.wait();

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Matrix multiplication took " << duration << " milliseconds." << std::endl;

    // The result is now in C
    // ...

    return 0;
}

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>

using namespace std;

// Function to multiply a row by a column
double multiplyRowByColumn(vector<vector<double>> &mat1, int row, vector<vector<double>> &mat2, int col) {
    double sum = 0;
    for (int i = 0; i < mat2.size(); i++) {
        sum += mat1[row][i] * mat2[i][col];
    }
    return sum;
}

// Function to perform matrix multiplication
void matmul(int start, int end, vector<vector<double>>& A, vector<vector<double>>& B, vector<vector<double>>& C) {
    for (int i = start; i < end; i++) {
        for (int j = 0; j < B[0].size(); j++) {
            C[i][j] = multiplyRowByColumn(A, i, B, j);
        }
    }
}

// Function to benchmark matrix multiplication
void benchmark_matmul(int M, int N, int K) {
    // Create random matrices A and B
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);

    vector<vector<double>> A(M, vector<double>(K));
    vector<vector<double>> B(K, vector<double>(N));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            A[i][j] = dis(gen);

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i][j] = dis(gen);

    // Result matrix C
    vector<vector<double>> C(M, vector<double>(N));

    // Perform multiplication and measure time
    auto start_time = chrono::high_resolution_clock::now();

    int numThreads = thread::hardware_concurrency();
    vector<thread> threads;
    int rowsPerThread = M / numThreads;

    for (int i = 0; i < numThreads; i++) {
        int start = i * rowsPerThread;
        int end = (i + 1) == numThreads ? M : (i + 1) * rowsPerThread;
        threads.emplace_back(matmul, start, end, ref(A), ref(B), ref(C));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = chrono::high_resolution_clock::now();

    // Calculate elapsed time in seconds
    chrono::duration<double> elapsed = end_time - start_time;

    // Calculate and print GFLOPS
    double gflops = 2.0 * M * N * K / (elapsed.count() * 1e9);
    cout << gflops << " GFLOP/s" << endl;
}

int main() {
    int M = 1024; // rows of A and C
    int N = 1024; // columns of B and C
    int K = 1024; // columns of A and rows of B

    // Benchmark matrix multiplication
    benchmark_matmul(M, N, K);

    return 0;
}

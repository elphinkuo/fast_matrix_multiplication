#include <iostream>
#include <vector>
#include <chrono>

typedef std::vector<std::vector<float>> Matrix;

// Function to generate a matrix filled with random numbers
Matrix generateRandomMatrix(int rows, int cols) {
    Matrix matrix(rows, std::vector<float>(cols, 0.0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    return matrix;
}

// Function to perform element-wise multiplication
Matrix elementWiseMultiplication(const Matrix& A, const Matrix& B) {
    int rows = A.size();
    int cols = A[0].size();
    Matrix C(rows, std::vector<float>(cols, 0.0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[i][j] = A[i][j] * B[i][j];
        }
    }
    return C;
}

// Function to benchmark the performance of element-wise multiplication
void benchmark(int M, int N) {
    Matrix A = generateRandomMatrix(M, N);
    Matrix B = generateRandomMatrix(M, N);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = elementWiseMultiplication(A, B);
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

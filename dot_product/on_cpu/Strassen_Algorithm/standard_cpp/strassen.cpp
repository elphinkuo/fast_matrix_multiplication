#include <iostream>
#include <vector>
#include <chrono>
#include <random>

typedef std::vector<std::vector<int>> Matrix;

Matrix add(const Matrix& a, const Matrix& b) {
    int n = a.size();
    Matrix result(n, std::vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

Matrix subtract(const Matrix& a, const Matrix& b) {
    int n = a.size();
    Matrix result(n, std::vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    return result;
}

Matrix strassen(const Matrix& a, const Matrix& b) {
    int n = a.size();
    if (n == 1) {
        return {{a[0][0] * b[0][0]}};
    }

    int half = n / 2;
    Matrix a11(half, std::vector<int>(half)), a12(half, std::vector<int>(half)),
            a21(half, std::vector<int>(half)), a22(half, std::vector<int>(half)),
            b11(half, std::vector<int>(half)), b12(half, std::vector<int>(half)),
            b21(half, std::vector<int>(half)), b22(half, std::vector<int>(half));

    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            a11[i][j] = a[i][j];
            a12[i][j] = a[i][j + half];
            a21[i][j] = a[i + half][j];
            a22[i][j] = a[i + half][j + half];
            b11[i][j] = b[i][j];
            b12[i][j] = b[i][j + half];
            b21[i][j] = b[i + half][j];
            b22[i][j] = b[i + half][j + half];
        }
    }

    Matrix p1 = strassen(a11, subtract(b12, b22));
    Matrix p2 = strassen(add(a11, a12), b22);
    Matrix p3 = strassen(add(a21, a22), b11);
    Matrix p4 = strassen(a22, subtract(b21, b11));
    Matrix p5 = strassen(add(a11, a22), add(b11, b22));
    Matrix p6 = strassen(subtract(a12, a22), add(b21, b22));
    Matrix p7 = strassen(subtract(a11, a21), add(b11, b12));

    Matrix c11 = add(subtract(add(p5, p4), p2), p6);
    Matrix c12 = add(p1, p2);
    Matrix c21 = add(p3, p4);
    Matrix c22 = add(subtract(add(p5, p1), p3), p7);

    Matrix c(n, std::vector<int>(n));
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            c[i][j] = c11[i][j];
            c[i][j + half] = c12[i][j];
            c[i + half][j] = c21[i][j];
            c[i + half][j + half] = c22[i][j];
        }
    }
    return c;
}

Matrix standard_multiplication(const Matrix& a, const Matrix& b) {
    int n = a.size();
    Matrix result(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

bool areMatricesEqual(const Matrix &a, const Matrix &b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (a[i][j] != b[i][j]) {
                return false;
            }
        }
    }

    return true;
}

void printMatrix(const Matrix &matrix) {
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << '\n';
    }
}



int main() {
    int n = 256;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    std::cout << "Generating two " << n << "x" << n << " matrices\n";

    Matrix a(n, std::vector<int>(n)), b(n, std::vector<int>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = dis(gen);
            b[i][j] = dis(gen);
        }
    }

    std::cout << "Running Strassen's Algorithm and Standard Multiplication\n";

    auto start = std::chrono::high_resolution_clock::now();
    Matrix result_strassen = strassen(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    auto strassen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    Matrix result_standard = standard_multiplication(a, b);
    end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Strassen's Algorithm took " << strassen_time << " milliseconds\n";
    std::cout << "Standard Multiplication took " << standard_time << " milliseconds\n";

    if (areMatricesEqual(result_strassen, result_standard)) {
        std::cout << "Results match\n";
    } else {
        std::cout << "Results do not match\n";
        std::cout << "Strassen's Algorithm Result:\n";
        printMatrix(result_strassen);
        std::cout << "Standard Multiplication Result:\n";
        printMatrix(result_standard);
    }

    return 0;
}

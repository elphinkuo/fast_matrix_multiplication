#include <iostream>
#include <vector>

using namespace std;

void WinogradMultiply(vector<vector<int>>& A, vector<vector<int>>& B, vector<vector<int>>& C, int n) {
    vector<int> row_factor(n), col_factor(n);
    
    // Precompute Row Factor for A
    for(int i = 0; i < n; i++) {
        row_factor[i] = A[i][0] * A[i][1];
        for(int j = 1; j < n/2; j++) {
            row_factor[i] = row_factor[i] + A[i][2 * j] * A[i][2 * j + 1];
        }
    }
    
    // Precompute Column Factor for B
    for(int i = 0; i < n; i++) {
        col_factor[i] = B[0][i] * B[1][i];
        for(int j = 1; j < n/2; j++) {
            col_factor[i] = col_factor[i] + B[2 * j][i] * B[2 * j + 1][i];
        }
    }

    // Initialize matrix C to zero
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C[i][j] = 0;
        }
    }

    // Compute the product
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C[i][j] = -row_factor[i] - col_factor[j];
            for(int k = 0; k < n / 2; k++) {
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j]);
            }
        }
    }

    // If n is odd, add the contributions of the odd rows and columns
    if(n % 2 != 0) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                C[i][j] += A[i][n - 1] * B[n - 1][j];
            }
        }
    }
}

int main() {
    // Define 3x3 matrices A, B and C
    vector<vector<int>> A = {{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9}};
    
    vector<vector<int>> B = {{10, 11, 12},
                             {13, 14, 15},
                             {16, 17, 18}};
    
    vector<vector<int>> C(3, vector<int>(3, 0));

    // Perform matrix multiplication
    WinogradMultiply(A, B, C, 3);

    // Print the result
    cout << "Result of Winograd matrix multiplication:" << endl;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}

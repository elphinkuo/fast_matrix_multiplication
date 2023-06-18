import numpy as np
from timeit import timeit

def matmul_python(C, A, B):
  for m in range(C.rows):
    for n in range(C.cols):
      for k in range(A.cols): C[m, n] += A[m, k] * B[k, n]

class Matrix:
  def __init__(self, value, rows, cols):
    self.value = value
    self.rows = rows
    self.cols = cols

  def __getitem__(self, idxs):  return self.value[idxs[0]][idxs[1]]
  def __setitem__(self, idxs, value): self.value[idxs[0]][idxs[1]] = value

def benchmark_matmul_python(M, N, K):
  A = Matrix(list(np.random.rand(M, K)), M, K)
  B = Matrix(list(np.random.rand(K, N)), K, N)
  C = Matrix(list(np.zeros((M, N))), M, N)

  # matmul_python operation time
  secs = timeit(lambda: matmul_python(C, A, B), number = 2)/2

  print('performance for mathmul_python')
  print(((2*M*N*K)/secs)/1e9, "GFLOP/s")

  A_ = np.random.rand(M, K)
  B_ = np.random.rand(M, K)
  # numpy operation time
  secs_ = timeit(lambda: np.matmul(A_, B_), number =2)/2
  print('performance for numpy matmul')
  print(((2*M*N*K)/secs_)/1e9, "GFLOP/s")

print('python code for 1024*1024 matrix would be super slow, for a quick view, can modify to 128*128 in the next line')
benchmark_matmul_python(1024, 1024, 1024)

# fast_matrix_multiplication
Different matrix multiplication implementation and benchmarking


A few months ago, I had the pleasure of tuning into the Modular AI 2023 product release keynote. It was a riveting experience, filled with anticipation and excitement, especially when the fast Matrix Multiplication section was presented by Jeremy Howard and Chris Lattner. My interest was so piqued that I immediately applied for the product preview, not just once, but several times. I'm still patiently awaiting approval.


During the interim, curiosity got the better of me, especially when it came to the fast matrix multiplication. I embarked on a journey to understand the feature better, and decided to take it a step further - I implemented a few versions of it myself and performed some benchmarks. This hands-on approach gave me a deeper insight into how fast matrix multiplication is implemented.


Within this repository, you'll find that I've conducted two types of matrix multiplication:
* The first being the __dot product__ of two matrices, both sized at __1024 * 1024__. I implemented this type using C++11 standard functions, C++11 using eigen library, C++11 using armadillo library, C++11 using SIMD AVX instuctions and Swift with Apple acceleration framework. The Swift with Apple acceleration framework amazingly accelerate the matrix multiplication to 532,550.639 times, using the same standard shown in Modular AI product release keynote.
* The second type is the element-wise multiplication, also known as the __Hadamard product__, of two matrices of the same size __(1024 * 1024)__.

The element-wise multiplication method, I must note, is particularly prevalent in Deep Learning models like Transformers.


## Dot product implementations


### Acceleration Results:
| Method           | GFLOP/s                       |  Acceleration (times)  |
| --------         | --------                      | ------         |
| Python 3.9       | 0.002536116 GFLOP/s |     1x         |
| C++11 standard   | 2.31342 GFLOP/s               |     912x       |
| C++11 eigen      | 0.521897 GFLOP/s              |     205.78x    |
| Numpy            | 87.856 GFLOP/s                |     34,709.7773x   |
| C++11 armadillo  | 107.782 GFLOP/s               |     __42,498.95x__ |
| C++11 SIMD AVX   | 
| Swift Acceleration| 1350.607 GFLOP/s             |     __532,550.639x__|


### Explaination:

* Python 3.9

  code in /dot_product/python_matmul, the first method is picked from and same as Modular AI product release keynote https://youtu.be/-3Kf2ZZU-dg, as benchmark base.
  
  
* C++11 standard

  code in /dot_product/standard_c++
  
  
* C++11 eigen

  code in /dot_product/eigen

* Numpy

  code in /dot_product/python_matmul, the second method.
  
  
* __C++11 Armadillo__ (42,498.95x acceleration)

  code in /dot_product/armadillo

  Here's a corresponding C++ version, written using the Armadillo library for linear algebra and other mathematical operations. Armadillo is well known for its high performance and efficient memory usage.

  This program creates a matrix class which uses Armadillo's fmat (floating point matrix) to store its data. matmul_native performs matrix multiplication using three nested loops, and benchmark_matmul_native measures the speed of this operation in GFLOPs (billions of floating-point operations per second).

  Please note that this is a simple version of the provided code and might not include all functionality that's implied by the Python code. Also, Armadillo has inbuilt high performance matrix multiplication which could be used instead of the native triple loop method.
  
  
* C++11 SIMD AVX

  code in /dot_product/simd_avx

  IMD stands for Single Instruction, Multiple Data. SIMD allows one single operation to be applied to a set of data simultaneously, which is extremely beneficial for tasks such as matrix multiplication, where the same operation (multiplication and addition) is applied over and over.

  For SIMD operations in C++, one can use the data types and intrinsic functions provided by the compiler. These will differ based on your hardware and the specific vector instruction set it supports, such as SSE, AVX, or AVX512 on Intel and AMD CPUs, or NEON on ARM CPUs.


  
* __Swift Acceleration__ (532,550.639x acceleration)

  code in /dot_product/swift

  In this Swift code, cblas_sgemm is a function from the Accelerate Framework that performs general matrix multiplication. The parameters specify how the matrices are stored in memory and the dimensions of the matrices.


  The cblas_sgemm function is part of Apple's Accelerate Framework, which is a collection of powerful and efficient numerical computing libraries. This function in particular is an implementation of general matrix multiply (GEMM), one of the fundamental operations of numerical linear algebra.


  The Accelerate Framework's version of this function is highly optimized for Apple hardware, using low-level programming and hardware-specific optimizations to achieve impressive speed. This can include the use of SIMD instructions, efficient cache usage, and other advanced techniques.


  Because of these optimizations, and the fact that the function is implemented in low-level code, it can be much faster than a similar function written in Swift (or other high-level languages) without such optimizations. This is why numerical computing libraries like Accelerate (or others like Intel's MKL, OpenBLAS, etc.) are widely used in scientific computing, data science, machine learning, and other fields that require heavy numerical computations.
  
  
  




## Hadamard product implementations




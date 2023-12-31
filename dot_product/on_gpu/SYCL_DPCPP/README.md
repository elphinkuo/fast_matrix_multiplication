SYCL defines abstractions to enable heterogeneous device programming, an important capability in the modern world which has not yet been solved directly in ISO C++. SYCL has evolved with the intent of influencing C++ direction around heterogeneous compute by creating productized proof points that can be considered in the context of C++ evolution.

A major goal of SYCL is to enable different heterogeneous devices to be used in a single application — for example simultaneous use of CPUs, GPUs, and FPGAs. Although optimized kernel code may differ across the architectures (since SYCL does not guarantee automatic and perfect performance portability across architectures), it provides a consistent language, APIs, and ecosystem in which to write and tune code for accelerator architectures. An application can coherently define variants of code optimized for architectures of interest, and can find and dispatch code to those architectures.

SYCL uses generic programming with templates and generic lambda functions to enable higher-level application software to be cleanly coded with optimized acceleration of kernel code across an extensive range of acceleration backend APIs, such as OpenCL and CUDA.

![Alt text](https://www.khronos.org/assets/uploads/apis/2022-sycl-diagram.jpg)

Above - creadit from: https://www.khronos.org/sycl/
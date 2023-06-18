# fast_matrix_multiplication
Different matrix multiplication implementation and benchmarking

A few months ago, I had the pleasure of tuning into the Modular AI 2023 product release keynote. It was a riveting experience, filled with anticipation and excitement, especially when the fast Matrix Multiplication section was presented by Jeremy Howard and Chris Lattner. My interest was so piqued that I immediately applied for the product preview, not just once, but several times. I'm still patiently awaiting approval.

During the interim, curiosity got the better of me, especially when it came to the fast matrix multiplication. I embarked on a journey to understand the feature better, and decided to take it a step further - I implemented a few versions of it myself and performed some benchmarks. This hands-on approach gave me a deeper insight into how fast matrix multiplication is implemented.

Within this repository, you'll find that I've conducted two types of matrix multiplication. The first being the dot product of two matrices, both sized at 1024 * 1024. The second type is the element-wise multiplication, also known as the Hadamard product, of two matrices of the same size (1023 * 1024). The element-wise multiplication method, I must note, is particularly prevalent in Deep Learning models like Transformers.

## Dot product implementations


## Hadamard product implementations




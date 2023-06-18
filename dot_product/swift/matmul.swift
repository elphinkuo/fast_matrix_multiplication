import Accelerate

let M = 1024
let N = 1024
let K = 1024

var A = (0..<M*K).map { _ in Float.random(in: 0..<1) }
var B = (0..<K*N).map { _ in Float.random(in: 0..<1) }
var C = [Float](repeating: 0.0, count: M*N)

let start = CFAbsoluteTimeGetCurrent()

cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N), Int32(K), 1, A, Int32(K), B, Int32(N), 0, &C, Int32(N))

let end = CFAbsoluteTimeGetCurrent()

let elapsed = end - start
let gflops = 2.0 * Double(M) * Double(N) * Double(K) / (elapsed * 1e9)
print("\(gflops) GFLOP/s")
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int N) {
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N || row < col) {
        if (row < N && col < N) C[row*N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    const unsigned start_k = col;
    const unsigned end_k = row;
    
    // Main summation loop with 4-step unrolling
    unsigned k = start_k;
    for (; k + 3 <= end_k; k += 4) {
        sum += __ldg(A + row*N + k) * __ldg(B + k*N + col);
        sum += __ldg(A + row*N + k+1) * __ldg(B + (k+1)*N + col);
        sum += __ldg(A + row*N + k+2) * __ldg(B + (k+2)*N + col);
        sum += __ldg(A + row*N + k+3) * __ldg(B + (k+3)*N + col);
    }
    for (; k <= end_k; ++k) {
        sum += __ldg(A + row*N + k) * __ldg(B + k*N + col);
    }

    // Warp-level reduction using shuffle instructions
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cg::this_thread_block());
    for (int offset = tile.size()/2; offset > 0; offset /= 2) {
        sum += tile.shfl_down(sum, offset);
    }

    if (tile.thread_rank() == 0) {
        C[row*N + col] = sum;
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    constexpr int threads = 16;
    const dim3 blocks((N + threads - 1)/threads, (N + threads - 1)/threads);
    triangular_mm_kernel<<<blocks, dim3(threads, threads)>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular mm with warp reduction");
}

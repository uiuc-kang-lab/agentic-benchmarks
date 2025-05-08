#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel assigns each valid lower-triangular element of the output matrix to one CUDA block.
// Each block performs a parallel reduction of the inner product sum over the index k for computing
// C[i, j] = sum_{k=j}^{i} A[i, k] * B[k, j].
// The reduction is done in two stages: first using shared memory for intra-block reduction, and then
// the final reduction stage uses warp-level primitives (__shfl_down_sync) for efficiency.
__global__ void triangular_mm_reduce_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    // Each block is responsible for one valid lower-triangular output element.
    // Use blockIdx.x as the linear index in the packed lower triangle.
    int idx = blockIdx.x;
    
    // Map the linear index 'idx' to the (i, j) coordinates in the lower triangular matrix.
    // Solve: i*(i+1)/2 <= idx < (i+1)*(i+2)/2. Using quadratic formula:
    float f = sqrtf(8.0f * idx + 1.0f);
    int i = (int)floorf((f - 1.0f) * 0.5f);
    int j = idx - (i * (i + 1)) / 2;

    // Ensure we are within bounds
    if (i >= N || j >= N) return;

    float sum = 0.0f;
    // Each thread in the block computes a partial sum for k from j to i.
    // The summation has (i - j + 1) terms.
    for (int k = j + threadIdx.x; k <= i; k += blockDim.x) {
        sum += A[i * N + k] * B[k * N + j];
    }

    // Allocate shared memory for reduction.
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Intra-block reduction using shared memory.
    // Reduce in powers of two until we get down to 32 threads.
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Use warp-level primitives for the final reduction within the last warp.
    float val = (threadIdx.x < 32) ? sdata[threadIdx.x] : 0.0f;
    // Reduce within a warp using __shfl_down_sync. The mask 0xffffffff covers all 32 lanes.
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // The first thread of the block writes the final result to the output matrix.
    if (threadIdx.x == 0) {
        C[i * N + j] = val;
    }
}

// C++ interface exposed to PyTorch.
// We preinitialize the output tensor C to all zeros so that upper triangular elements are 0.
// Then we launch the kernel for only the valid lower triangular elements (i >= j).
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);

    // Create output tensor and initialize it with zeros, ensuring upper triangular elements are 0.
    auto C = torch::zeros_like(A);

    // Number of valid lower triangular elements.
    int num_elements = N * (N + 1) / 2;

    // Define block size for reduction.
    const int blockSize = 128;
    // Each block computes one output element, so grid dimension equals the number of lower triangular elements.
    dim3 grid(num_elements);
    dim3 block(blockSize);

    // Dynamic shared memory size: blockSize * sizeof(float).
    size_t sharedMemSize = blockSize * sizeof(float);

    triangular_mm_reduce_kernel<<<grid, block, sharedMemSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with shared memory and warp-level reduction (CUDA)");
}

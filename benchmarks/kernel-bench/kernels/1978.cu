#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define segment tile size and block size for reduction
#define TS 32
#define BLOCK_SIZE 128

// Kernel that partitions the computation of each output (i, j) (for i>=j) into segments along k
// Each block computes the partial dot product over a segment of the summation index and then updates
// the global result using a single atomicAdd. This minimizes atomic usage and global contention.
__global__ void triangular_mm_kernel_atomic(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {

    // Map grid dimensions: 
    //   blockIdx.y: row index i
    //   blockIdx.x: column index j
    //   blockIdx.z: segment index for the k dimension
    int i = blockIdx.y;  // row index
    int j = blockIdx.x;  // column index
    int seg = blockIdx.z; // segment index for the summation over k
    
    // Only compute lower triangular part
    if(i < j) return;

    // For output element C[i,j], the summation index k runs from j to i inclusive
    int L = i - j + 1;  // total number of terms
    
    // Each segment covers TS elements in the k-dimension. Compute segment starting index
    int seg_start = j + seg * TS;
    if(seg_start >= (i + 1)) return; // this segment exceeds the valid k range

    // Compute segment end within valid limits
    int seg_end = min(i, seg_start + TS - 1);

    // Each block uses BLOCK_SIZE threads to compute partial sum over indices [seg_start, seg_end]
    float partial = 0.f;
    int tid = threadIdx.x;
    
    // Each thread striding over the segment
    for (int k = seg_start + tid; k <= seg_end; k += blockDim.x) {
        partial += A[i * N + k] * B[k * N + j];
    }

    // Reduction in shared memory
    __shared__ float sdata[BLOCK_SIZE];
    sdata[tid] = partial;
    __syncthreads();

    // Standard reduction loop
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Thread 0 from this block adds the block's partial sum into the output using atomicAdd
    if(tid == 0) {
        atomicAdd(&C[i * N + j], sdata[0]);
    }
}

// Host interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    // Initialize output C to zero so that atomic adds accumulate correctly
    auto C = torch::zeros_like(A);

    // Grid dimensions: 
    //   grid.x: j index (from 0 to N-1)
    //   grid.y: i index (from 0 to N-1)
    //   grid.z: segments along k - maximum segments = ceil(N/TS)
    dim3 grid(N, N, (N + TS - 1) / TS);
    int threads = BLOCK_SIZE;

    triangular_mm_kernel_atomic<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication using atomic reduction (CUDA)");
}

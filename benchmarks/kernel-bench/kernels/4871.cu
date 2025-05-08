#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel using shared memory for warp-level partial sums and reduction
__global__ void l1_norm_forward_kernel_shared(const float* __restrict__ x,
                                                float* __restrict__ out,
                                                int N,
                                                int D) {
    extern __shared__ float sdata[];  // Shared memory for partial sums
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockDimVal = blockDim.x;

    // Each thread processes multiple elements in the row
    float sum = 0.0f;
    for (int col = tid; col < D; col += blockDimVal) {
        float val = x[row * D + col];
        sum += fabsf(val);
    }

    // Perform warp-level reduction on the partial sum
    sum = warpReduceSum(sum);

    // Write the reduced value of each warp to shared memory
    if ((tid & (warpSize - 1)) == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();

    // Final reduction: let first warp reduce the warp sums stored in shared memory
    float total = 0.0f;
    int numWarps = (blockDimVal + warpSize - 1) / warpSize;
    if (tid < numWarps) {
        total = sdata[tid];
        total = warpReduceSum(total);
    }
    
    // Ensure that a zero sum is avoided by the first thread
    if (tid == 0) {
        if (total == 0.0f)
            total = 1e-12f;
        sdata[0] = total;
    }
    __syncthreads();
    total = sdata[0];

    // Normalize each element of the row
    for (int col = tid; col < D; col += blockDimVal) {
        float val = x[row * D + col];
        out[row * D + col] = val / total;
    }
}

// Host function interfaced with PyTorch
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA.");
    TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
    x = x.contiguous();

    auto out = torch::empty_like(x);
    int N = x.size(0);
    int D = x.size(1);

    int threads = std::min<int>(1024, D); // Limit threads to 1024 or D, whichever is smaller
    int numWarps = (threads + 31) / 32;
    int shared_mem_size = numWarps * sizeof(float);

    l1_norm_forward_kernel_shared<<<N, threads, shared_mem_size>>>(x.data_ptr<float>(),
                                                                     out.data_ptr<float>(),
                                                                     N, D);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L1 Normalization forward pass (CUDA with shared memory optimization)");
}

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);   \
    CHECK_CONTIGUOUS(x)

// Hybrid kernel: Use vectorized loads (float4) for the bulk of the data and
// shared memory to handle any tail (non-multiple-of-4) elements in a coalesced way.
__global__ void elu_kernel_hybrid(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   float alpha,
                                   int n,   // total number of elements
                                   int n4)  // number of float4 elements in the bulk
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // ----- Process main, vectorized part -----
    // Each thread works on 4 elements at a time using float4 if in bounds
    if (tid < n4) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* out4 = reinterpret_cast<float4*>(out);
        float4 in_val = x4[tid];
        float4 res;
        res.x = (in_val.x > 0.f) ? in_val.x : alpha * (expf(in_val.x) - 1.f);
        res.y = (in_val.y > 0.f) ? in_val.y : alpha * (expf(in_val.y) - 1.f);
        res.z = (in_val.z > 0.f) ? in_val.z : alpha * (expf(in_val.z) - 1.f);
        res.w = (in_val.w > 0.f) ? in_val.w : alpha * (expf(in_val.w) - 1.f);
        out4[tid] = res;
    }

    // ----- Process tail elements -----
    // If n is not a multiple of 4, handle the remaining elements using shared memory
    int tail_start = n4 * 4;  // starting index for tail elements
    int total_threads = gridDim.x * blockDim.x;

    // Declare shared memory buffer. Each block allocates blockDim.x floats.
    extern __shared__ float tile[];

    // Use a grid-stride loop over the tail portion
    for (int i = tail_start + tid; i < n; i += total_threads) {
        int local_idx = threadIdx.x;  // each thread's slot in shared memory

        // Load scalar element from global memory into shared memory
        // (Even though for one element, using shared memory here mimics kernel1's approach
        // for coalesced memory accesses on a tile—even for tail elements.)
        tile[local_idx] = x[i];
        __syncthreads();

        // Perform the ELU activation
        float val = tile[local_idx];
        float result = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
        __syncthreads();

        // Write the result back to global memory
        out[i] = result;
        __syncthreads();  // Ensure that shared memory is ready for next iteration
    }
}

// Interface function called from Python
torch::Tensor elu_cuda_hybrid(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Calculate the number of complete float4 groups
    int n4 = n / 4;  // main vectorized part covers n4 * 4 elements

    // Determine tail count (n % 4)
    int tail_count = n - n4 * 4;

    // Decide grid size: we want to cover both the vectorized and tail parts
    // (n4 is typically >> tail_count, so using n4 works in most cases)
    int thread_needed = (n4 > tail_count ? n4 : tail_count);
    const int threads = 256;
    int blocks = (thread_needed + threads - 1) / threads;

    // Allocate shared memory per block for tail processing
    size_t sharedMemSize = threads * sizeof(float);

    elu_kernel_hybrid<<<blocks, threads, sharedMemSize>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        n,
        n4
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_hybrid, "Hybrid ELU Activation with vectorized load and shared memory tail handling (CUDA)");
}

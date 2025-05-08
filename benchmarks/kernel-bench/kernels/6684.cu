#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

// Perform warp-level product reduction in a branch-free manner using shuffle intrinsics
__inline__ __device__ float warpReduceProduct(float val) {
    // Unrolled reduction loop without divergent branching
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that performs product reduction for each output element
// It minimizes warp divergence by organizing the reduction into two stages with uniform control flow
__global__ void prod_reduction_kernel(const float* __restrict__ input, float* __restrict__ output, int dim_size, int stride) {
    // Each block is responsible for one output element
    int outIdx = blockIdx.x;
    int tid = threadIdx.x;

    // Each thread computes a partial product over a strided segment of the reduction dimension
    float prod = 1.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        prod *= input[outIdx + i * stride];
    }

    // Stage 1: Intra-warp reduction using warp shuffle, which avoids divergent branches within the warp
    float warpProd = warpReduceProduct(prod);

    // Each warp obtains a single reduced value. Use shared memory to collect warp results.
    // To minimize divergence, we broadcast the warp result uniformly within each warp.
    __shared__ float warpResults[32]; // maximum warps per block (assuming blockDim.x is a multiple of 32)
    int warpId = tid / WARP_SIZE;
    int lane = tid & (WARP_SIZE - 1);

    // Instead of a divergent if, we use shuffle so all threads in the warp obtain the same value
    float broadcastWarpProd = __shfl_sync(0xffffffff, warpProd, 0);
    // Only one representative per warp writes the value; extra writes are avoided by a simple branch
    if (lane == 0) {
        warpResults[warpId] = broadcastWarpProd;
    }
    __syncthreads();

    // Stage 2: Reduction across warp results by the first warp
    // Determine the number of warps in the block (assumed blockDim.x is a multiple of 32)
    int numWarps = blockDim.x / WARP_SIZE;
    // Each thread in the first warp loads a warp result if available; else uses the multiplicative identity
    // To refactor the conditional logic, we use a ternary operator that is often optimized into predicated code
    float val = (tid < numWarps) ? warpResults[tid] : 1.0f;
    val = warpReduceProduct(val);

    // The first thread writes the final result
    if (tid == 0) {
        output[outIdx] = val;
    }
}

// Forward function for product reduction over a specified dimension
// It removes the reduced dimension from the output shape and launches one block per output element
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);
    
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());
    
    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch configuration: one block per output element, 256 threads per block (multiple of warp size)
    int threads = 256;
    int blocks = num_elements;
    size_t shared_mem_size = 32 * sizeof(float); // shared memory for up to 32 warp results

    prod_reduction_kernel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA) with branch-free warp reduction");
}

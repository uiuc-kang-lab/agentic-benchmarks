#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Warp-level reduction function using shuffle for product reduction
__device__ __forceinline__ float warpReduceProd(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that uses __ldg() for read-only accesses and vectorized loads for 128-bit aligned memory
__global__ void ldg_aligned_prod_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int dim_size,
                                         int stride) {
    __shared__ float warp_results[NUM_WARPS];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;  // Each block processes one output element
    int lane_id = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;
    
    float local_prod = 1.0f;

    // If the reduction dimension is contiguous and its size is a multiple of 4, use 128-bit vectorized loads
    if (stride == 1 && (dim_size % 4 == 0)) {
        // Reinterpret input pointer as float4 pointer for 128-bit accesses
        const float4* input_vec = reinterpret_cast<const float4*>(input);
        int vec_length = dim_size / 4;
        // Each block's segment starts at index: bid * vec_length
        // Each thread processes multiple float4 elements in a strided loop
        for (int i = tid; i < vec_length; i += BLOCK_SIZE) {
            float4 data = __ldg(&input_vec[bid * vec_length + i]);
            // Multiply the four float components
            local_prod *= (data.x * data.y * data.z * data.w);
        }
    } else {
        // Fallback: use scalar __ldg() loads for non-contiguous or non-vectorizable cases
        for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
            local_prod *= __ldg(&input[bid + i * stride]);
        }
    }
    
    // Warp-level reduction using shuffle instructions
    local_prod = warpReduceProd(local_prod);
    
    // Write warp-level results to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = local_prod;
    }
    __syncthreads();
    
    // Final reduction performed by the first warp
    if (warp_id == 0) {
        float warp_prod = (lane_id < NUM_WARPS) ? warp_results[lane_id] : 1.0f;
        warp_prod = warpReduceProd(warp_prod);
        if (lane_id == 0) {
            output[bid] = warp_prod;
        }
    }
}

// Forward function for the Torch extension
// Performs product reduction over the specified dimension using the optimized kernel

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
    
    // Launch one block per output element
    int threads = BLOCK_SIZE;
    int blocks = num_elements;
    
    ldg_aligned_prod_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LDG-based aligned product reduction over a dimension (CUDA)");
}

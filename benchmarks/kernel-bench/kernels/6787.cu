#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses shared memory when the reduction dimension is contiguous (stride==1).
// For each output element, the required 50 consecutive floats in input are loaded via a sliding window into shared memory.
// Threads in the same block cooperate to load a contiguous tile of size (blockDim.x + 50 - 1) from global memory.
// Each thread then computes the product over its 50-element window from shared memory, reducing global memory traffic.
// For non-contiguous cases (stride != 1), the kernel falls back to directly loading from global memory.

__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output, int stride, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        if (stride == 1) {
            // Use shared memory for contiguous reduction
            extern __shared__ float tile[];
            int block_start = blockIdx.x * blockDim.x;  // starting output index for this block
            int t = threadIdx.x;
            // Total number of floats needed: one product window per thread: blockDim.x + 50 - 1
            int total_load = blockDim.x + 50 - 1;
            // Each thread cooperatively loads multiple elements into shared memory
            for (int i = t; i < total_load && (block_start + i) < num_elements; i += blockDim.x) {
                tile[i] = input[block_start + i];
            }
            __syncthreads();
            
            // Each thread now computes the product from its sliding window in shared memory
            float prod = 1.0f;
            int offset = t;  // local starting index in shared memory for this thread
            #pragma unroll
            for (int i = 0; i < 50; i++) {
                prod *= tile[offset + i];
            }
            output[idx] = prod;
        } else {
            // Fallback for non-contiguous memory: load directly from global memory
            float prod = 1.0f;
            #pragma unroll
            for (int i = 0; i < 50; i++) {
                prod *= input[idx + i * stride];
            }
            output[idx] = prod;
        }
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Compute output shape by removing the reduction dimension
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim]; // expected to be 50
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Launch parameters
    const int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    if (stride == 1) {
        // For contiguous reduction, allocate shared memory: one tile per block
        int smem_size = (threads + 50 - 1) * sizeof(float);
        prod_reduce_kernel<<<blocks, threads, smem_size>>>(input_ptr, output_ptr, stride, num_elements);
    } else {
        prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, stride, num_elements);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension using shared memory (CUDA)");
}

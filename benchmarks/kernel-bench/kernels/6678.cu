#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Custom atomic multiplication for floats using atomicCAS
__device__ float atomicMul(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float old_f = __int_as_float(assumed);
        float new_f = old_f * val;
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_f));
    } while (assumed != old);
    return __int_as_float(old);
}

// Kernel: each block processes a chunk of the reduction dimension for a given output element
__global__ void prod_reduce_kernel_atomic(const float* input, float* output, int dim_size, int stride, int chunk_size) {
    // Determine the output element index (flattened index) this block contributes to
    int outIdx = blockIdx.x;      
    // blockIdx.y indicates which chunk along the reduced dimension this block will handle
    int chunk = blockIdx.y;
    int start = chunk * chunk_size;
    int end = start + chunk_size;
    if (end > dim_size) end = dim_size;

    int tid = threadIdx.x;
    float partial = 1.0f;
    
    // Each thread processes a part of the chunk in a strided manner
    for (int i = start + tid; i < end; i += blockDim.x) {
         partial *= input[outIdx + i * stride];
    }
    
    // Reduction within the block using shared memory
    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    
    // Tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Use atomic multiplication to update the final output for this element
    if (tid == 0) {
       atomicMul(&output[outIdx], sdata[0]);
    }
}

// Forward function for product reduction over a given dimension
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    // Get input sizes and determine the size of the reduction dimension
    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    // Remove the reduction dimension for the output shape
    sizes.erase(sizes.begin() + dim);
    // Initialize output tensor with ones (multiplicative identity) so that atomicMul works correctly
    torch::Tensor output = torch::ones(sizes, x.options());

    int num_output_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Parameters for parallel reduction
    int threads = 256;
    int chunk_size = 1024; // Tunable parameter: number of elements per reduction chunk
    int grid_y = (dim_size + chunk_size - 1) / chunk_size;

    // Configure grid: one dimension for each output element and one for chunks along the reduction axis
    dim3 blocks(num_output_elements, grid_y);

    // Launch kernel with dynamic shared memory for intra-block reduction
    prod_reduce_kernel_atomic<<<blocks, threads, threads * sizeof(float)>>>(
        input_ptr, output_ptr, dim_size, stride, chunk_size);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (Optimized CUDA with atomic operations)");
}

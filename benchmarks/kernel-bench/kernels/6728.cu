#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_phase1_kernel(const float* input, float* partial_results, 
                                        int dim_size, int stride, int num_elements) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = blockDim.x * bid + tid;
    
    // Initialize shared memory
    shared_mem[tid] = 1.0f;
    
    if (gid < num_elements) {
        float product = 1.0f;
        int base_idx = (gid / stride) * stride * dim_size + (gid % stride);
        
        // Compute partial product for this thread
        for (int i = 0; i < dim_size; ++i) {
            product *= input[base_idx + i * stride];
        }
        shared_mem[tid] = product;
    }
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && gid + s < num_elements) {
            shared_mem[tid] *= shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global mem
    if (tid == 0) {
        partial_results[bid] = shared_mem[0];
    }
}

__global__ void prod_reduce_phase2_kernel(float* partial_results, float* output, 
                                        int num_blocks, int num_elements) {
    const int tid = threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    
    for (int idx = tid; idx < num_elements; idx += total_threads) {
        float final_product = 1.0f;
        int start_block = idx * num_blocks;
        int end_block = min((idx + 1) * num_blocks, num_blocks);
        
        for (int b = start_block; b < end_block; ++b) {
            final_product *= partial_results[b];
        }
        output[idx] = final_product;
    }
}

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

    const int threads = 256;
    const int blocks = std::min(256, (num_elements + threads - 1) / threads);
    
    // Allocate temporary storage for partial results
    auto partial_results = torch::empty({blocks}, x.options());
    float* partial_results_ptr = partial_results.data_ptr<float>();

    // Launch phase 1: parallel reduction within blocks
    prod_reduce_phase1_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input_ptr, partial_results_ptr, dim_size, stride, num_elements
    );

    // Launch phase 2: combine partial results
    const int phase2_blocks = std::min(32, (num_elements + threads - 1) / threads);
    prod_reduce_phase2_kernel<<<phase2_blocks, threads>>>(
        partial_results_ptr, output_ptr, blocks, num_elements
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}
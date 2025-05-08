#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_kernel(const float* input, float* output, int dim_size, int stride, int num_elements) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int grid_stride = blockDim.x * gridDim.x;
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    
    // Ensure mapping of threads covers all elements with grid-level stride
    for (int idx = tid; idx < num_elements; idx += grid_stride) {
        float product = 1.0f;
        
        // Unroll small iterations for better instruction-level parallelism
        #pragma unroll 4
        for (int i = 0; i < dim_size; ++i) {
            product *= input[idx + i * stride];
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll 5
        for (int offset = 16; offset > 0; offset /= 2) {
            product *= __shfl_down_sync(0xffffffff, product, offset);
        }
        
        // First thread in each warp writes to shared memory
        if (lane_id == 0) {
            shared_mem[warp_id] = product;
        }
        
        __syncthreads();
        
        // Final reduction across warps (done by first warp)
        if (warp_id == 0 && lane_id < (blockDim.x / 32)) {
            product = shared_mem[lane_id];
            
            // Final warp reduction
            #pragma unroll 5
            for (int offset = 16; offset > 0; offset /= 2) {
                product *= __shfl_down_sync(0xffffffff, product, offset);
            }
            
            if (lane_id == 0) {
                output[blockIdx.x] = product;
            }
        }
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
    const int blocks = (num_elements + threads - 1) / threads;

    prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}
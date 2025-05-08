#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void balanced_prod_reduce_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int dim_size,
                                             int stride,
                                             int num_elements) {
    extern __shared__ float shared_input[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Ensure workload is evenly distributed by using a grid-stride loop
    for (int idx = tid; idx < num_elements; idx += total_threads) {
        float product = 1.0f;
        
        // Process elements in chunks to better utilize shared memory
        const int CHUNK_SIZE = 32;  // Process 32 elements at a time
        for (int base = 0; base < dim_size; base += CHUNK_SIZE) {
            // Load chunk into shared memory
            int remaining = min(CHUNK_SIZE, dim_size - base);
            if (threadIdx.x < remaining) {
                shared_input[threadIdx.x] = input[idx + (base + threadIdx.x) * stride];
            }
            __syncthreads();
            
            // Process chunk with partial loop unrolling
            #pragma unroll 4
            for (int i = 0; i < remaining; ++i) {
                product *= shared_input[i];
            }
            __syncthreads();
        }
        output[idx] = product;
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
    dim3 blocks((num_elements + threads - 1) / threads);

    balanced_prod_reduce_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced workload product reduction over a dimension (CUDA)");
}

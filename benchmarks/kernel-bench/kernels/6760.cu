#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Custom atomic multiplication for floats implemented using atomicCAS
__device__ float atomicMul(float* address, float val) {
    int* addr_as_int = (int*) address;
    int old_int = *addr_as_int;
    int assumed_int;
    do {
        assumed_int = old_int;
        float old_float = __int_as_float(assumed_int);
        float new_val = old_float * val;
        int new_int = __float_as_int(new_val);
        old_int = atomicCAS(addr_as_int, assumed_int, new_int);
    } while (assumed_int != old_int);
    return __int_as_float(old_int);
}


__global__ void prod_reduce_kernel_optimized(const float* input, float* output, int dim_size, int stride, int num_elements) {
    extern __shared__ float shared_data[];
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;

    float product = 1.0f;
    if (global_idx < num_elements) {
        for (int i = 0; i < dim_size; ++i) {
            product *= input[global_idx + i * stride];
        }
    }
    shared_data[local_idx] = product;
    __syncthreads();

    if (local_idx == 0) {
        float block_product = 1.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            block_product *= shared_data[i];
        }
        atomicMul(&output[blockIdx.x], block_product);  // Use atomic operation at block level
    }
}

torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::ones(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    size_t shared_memory_size = threads * sizeof(float);
    prod_reduce_kernel_optimized<<<blocks, threads, shared_memory_size>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}

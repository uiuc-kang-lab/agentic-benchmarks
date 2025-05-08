#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void prod_reduce_kernel(const float* input, float* output, int dim_size, int stride, int num_elements) {
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int total_threads = blockDim.x * gridDim.x;

    // Accumulate the product in shared memory (minimize global memory accesses)
    extern __shared__ float sdata[];
    sdata[tid] = 1.0f;
    for (int idx = global_tid; idx < num_elements; idx += total_threads) {
        float product = 1.0f;
        for (int i = 0; i < dim_size; ++i) {
            product *= input[idx + i * stride];
        }
        atomicMul(&sdata[tid], product);
    }
    __syncthreads();

    // Reduce to the first location for each block
    if (tid == 0) {
        for (int i = 1; i < blockDim.x; ++i) {
            sdata[0] *= sdata[i];
        }
        atomicMul(&output[blockIdx.x], sdata[0]);
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
    int sharedMem = threads * sizeof(float);

    prod_reduce_kernel<<<blocks, threads, sharedMem>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension with optimized atomic operations (CUDA)");
}
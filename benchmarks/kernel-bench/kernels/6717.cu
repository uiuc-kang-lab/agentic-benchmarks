#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Optimized kernel: each block handles one output element reduction over the specified dimension
__global__ void optimized_prod_reduce_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int dim_size,
                                              int stride,
                                              int num_elements) {
    int out_idx = blockIdx.x;
    if (out_idx >= num_elements) return;
    int tid = threadIdx.x;

    // Each thread computes a partial product over the reduction dimension
    float partial_prod = 1.0f;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        partial_prod *= input[out_idx + i * stride];
    }

    // Use shared memory for block-level reduction
    extern __shared__ float sdata[];
    sdata[tid] = partial_prod;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the final product from thread 0 of each block
    if (tid == 0) {
        output[out_idx] = sdata[0];
    }
}

// Host function to launch the optimized kernel
torch::Tensor forward(torch::Tensor x, int dim) {
    CHECK_INPUT(x);

    auto sizes = x.sizes().vec();
    int dim_size = sizes[dim];
    sizes.erase(sizes.begin() + dim);
    torch::Tensor output = torch::empty(sizes, x.options());

    int num_elements = output.numel();
    int stride = x.stride(dim);

    // Launch one block per output element with 256 threads per block
    const int threads = 256;
    dim3 blocks(num_elements);
    int sharedMem = threads * sizeof(float);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    optimized_prod_reduce_kernel<<<blocks, threads, sharedMem>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized product reduction over a dimension (CUDA)");
}

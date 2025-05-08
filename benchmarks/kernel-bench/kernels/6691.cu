#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32

// Optimized kernel combining ideas from both implementations, focusing on minimizing memory transactions
// and maximizing computational efficiency by unrolling the loop and using the shuffle reduction technique.
__global__ void optimized_prod_reduction_kernel(const float* input, float* output, int dim_size, int stride, int num_elements) {
    int outIdx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    // Unrolling the loop for performance and reducing the number of transactions
    float partial = 1.0f;
    int step = blockDim.x * gridDim.x;
    for (int i = tid; i < dim_size; i += step) {
        partial *= input[outIdx + i * stride];
    }
    sdata[tid] = partial;
    __syncthreads();

    // Shared memory reduction
    for (int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    // warp-level reduction
    if (tid < WARP_SIZE) {
        float warpVal = sdata[tid];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            warpVal *= __shfl_down_sync(0xffffffff, warpVal, offset);
        }
        if (tid == 0) {
            output[outIdx] = warpVal;
        }
    }
}

// Forward function
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

    int threads = 256;
    int blocks = num_elements;
    size_t shared_mem_size = threads * sizeof(float);

    optimized_prod_reduction_kernel<<<blocks, threads, shared_mem_size>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Product reduction over a dimension (CUDA)");
}
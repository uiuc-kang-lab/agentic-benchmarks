#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void combined_prod_reduce_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int dim_size,
                                             int stride,
                                             int num_elements) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Use a single loop to manage both block and thread level reductions
    for (int idx = out_idx; idx < num_elements; idx += total_threads) {
        float product = 1.0f;
        for (int i = 0; i < dim_size; i += blockDim.x) {
            if (threadIdx.x + i < dim_size) {
                product *= input[idx + (threadIdx.x + i) * stride];
            }
        }

        extern __shared__ float sdata[];
        sdata[threadIdx.x] = product;
        __syncthreads();

        // Parallel reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] *= sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[idx / blockDim.x] = sdata[0];
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

    const int threads = 256;
    dim3 blocks((num_elements + threads - 1) / threads);
    int sharedMem = threads * sizeof(float);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    combined_prod_reduce_kernel<<<blocks, threads, sharedMem>>>(input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined optimized product reduction over a dimension (CUDA)");
}

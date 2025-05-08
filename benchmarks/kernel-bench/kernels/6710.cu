#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ __forceinline__ float warpReduceProd(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void shfl_down_optimized_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int dim_size,
                                           int stride) {
    __shared__ float shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Load data into shared memory
    float local_prod = 1.0f;
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        local_prod *= input[bid + i * stride];
    }
    shared_data[tid] = local_prod;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_data[tid] *= shared_data[tid + offset];
        }
        __syncthreads();
    }

    // Final warp-level reduction
    if (warp_id == 0 && lane_id == 0) {
        output[bid] = shared_data[0];
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

    dim3 blocks(num_elements);
    dim3 threads(BLOCK_SIZE);

    shfl_down_optimized_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized product reduction with warp-level primitives (CUDA)");
}
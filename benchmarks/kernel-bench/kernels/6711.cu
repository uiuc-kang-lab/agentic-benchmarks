#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void stride_loop_prod_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        const int dim_size,
                                        const int stride) {
    __shared__ float shared_data[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;

    // Initialize partial product
    float thread_prod = 1.0f;

    // Use stride loop to handle workloads larger than the number of threads
    for (int i = tid; i < dim_size; i += BLOCK_SIZE) {
        thread_prod *= input[bid + i * stride];
    }

    // Store partial results in shared memory
    shared_data[tid] = thread_prod;
    __syncthreads();

    // Reduce within block using shared memory
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_data[tid] *= shared_data[tid + offset];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
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

    // Launch configuration
    int threads = BLOCK_SIZE;
    int blocks = num_elements;

    stride_loop_prod_kernel<<<blocks, threads>>>(input_ptr, output_ptr, dim_size, stride);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride loop product reduction (CUDA)");
}
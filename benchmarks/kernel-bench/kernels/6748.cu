#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__inline__ __device__ float warpReduceProduct(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_prod_reduce_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int dim_size,
                                      int stride,
                                      int num_elements) {
    extern __shared__ float sdata[];
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int warp_id = tid / WARP_SIZE;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;

    for (int idx = gid; idx < num_elements; idx += grid_stride) {
        // Compute initial product for this thread
        float thread_prod = 1.0f;
        for (int i = 0; i < dim_size; i++) {
            thread_prod *= input[idx + i * stride];
        }

        // First warp-level reduction
        float warp_prod = warpReduceProduct(thread_prod);

        // Write reduced warp results to shared memory
        if (lane_id == 0) {
            sdata[warp_id] = warp_prod;
        }
        __syncthreads();

        // Final reduction using first warp
        if (warp_id == 0) {
            float val = (tid < (blockDim.x / WARP_SIZE)) ? sdata[tid] : 1.0f;
            val = warpReduceProduct(val);
            if (lane_id == 0) {
                output[idx] = val;
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

    const int threads = 256;
    const int blocks = std::min(256, (num_elements + threads - 1) / threads);
    const int shared_mem_size = (threads / WARP_SIZE) * sizeof(float);

    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    warp_prod_reduce_kernel<<<blocks, threads, shared_mem_size>>>(
        input_ptr, output_ptr, dim_size, stride, num_elements);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized product reduction (CUDA)");
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__inline__ __device__ float blockReduceSum(float val) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Warp-level reduction
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += warp.shfl_down(val, offset);
    }

    // Block-level reduction
    if (warp.thread_rank() == 0) shared[wid] = val;
    block.sync();

    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            val += warp.shfl_down(val, offset);
        }
    }
    return val;
}

__global__ void instance_norm_kernel_coalesced(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    extern __shared__ float temp_storage[];

    int instance_id = blockIdx.x;
    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_ptr[i];
        temp_storage[i] = val;
        sum += val;
        sum_sq += val * val;
    }

    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float mean;
    __shared__ float invstd;
    if (threadIdx.x == 0) {
        mean = sum / HW;
        float var = (sum_sq / HW) - (mean * mean);
        var = (var < 0.f) ? 0.f : var;
        invstd = rsqrtf(var + eps);
    }
    __syncthreads();

    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = temp_storage[i];
        y_ptr[i] = ((val - mean) * invstd) * scale + shift;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined() && weight.numel() > 0)
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined() && bias.numel() > 0)
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto y = torch::empty_like(x);

    int HW = H * W;
    int block_size = 256; // Fixed block size for simplicity
    int blocks = N * C;
    int shared_mem_size = HW * sizeof(float);

    instance_norm_kernel_coalesced<<<blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA)");
}

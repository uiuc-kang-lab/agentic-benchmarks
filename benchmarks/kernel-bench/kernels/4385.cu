#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    float eps,
    int instances_per_block
) {
    extern __shared__ float shared_data[];
    float* shared_mean = shared_data;
    float* shared_var = &shared_data[instances_per_block];

    const int HW = H * W;
    const int instance_idx = blockIdx.x * instances_per_block;
    const int local_instance_idx = threadIdx.x / (HW / instances_per_block + 1);
    
    if (local_instance_idx >= instances_per_block || 
        instance_idx + local_instance_idx >= N * C) return;

    const int n = (instance_idx + local_instance_idx) / C;
    const int c = (instance_idx + local_instance_idx) % C;
    
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    const int local_tid = threadIdx.x % (HW / instances_per_block + 1);
    for (int i = local_tid; i < HW; i += HW / instances_per_block + 1) {
        float val = x_ptr[i];
        sum += val;
        sq_sum += val * val;
    }

    sum = blockReduceSum(sum);
    sq_sum = blockReduceSum(sq_sum);

    if (local_tid == 0) {
        float mean = sum / HW;
        float var = sq_sum / HW - mean * mean;
        var = (var < 0.f) ? 0.f : var;
        shared_mean[local_instance_idx] = mean;
        shared_var[local_instance_idx] = var;
    }
    __syncthreads();

    const float mean = shared_mean[local_instance_idx];
    const float var = shared_var[local_instance_idx];
    const float inv_std = rsqrtf(var + eps);

    for (int i = local_tid; i < HW; i += HW / instances_per_block + 1) {
        float normalized = (x_ptr[i] - mean) * inv_std;
        if (weight && bias) {
            normalized = normalized * weight[c] + bias[c];
        }
        y_ptr[i] = normalized;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined()) TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");

    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    int HW = H * W;

    auto y = torch::empty_like(x);

    const int threads = 256;
    int instances_per_block = HW <= 256 ? 8 : (HW <= 1024 ? 4 : 1);
    int blocks = (N * C + instances_per_block - 1) / instances_per_block;
    
    size_t shared_mem_size = 2 * instances_per_block * sizeof(float);

    instance_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps),
        instances_per_block
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA)");
}
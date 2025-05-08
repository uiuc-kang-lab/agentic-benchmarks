#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
#else
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
#endif
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);

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
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    extern __shared__ float shared_data[];
    float* temp_storage = shared_data;
    
    int instance_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (instance_id >= N * C) return;
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
    
    __shared__ float mean_sh;
    __shared__ float invstd_sh;
    
    if (threadIdx.x == 0) {
        mean_sh = sum / HW;
        float var = __fmaf_rn(-mean_sh, mean_sh, sum_sq / HW);
        var = (var < 0.f) ? 0.f : var;
        invstd_sh = rsqrtf(var + eps);
    }
    __syncthreads();
    
    float scale = weight ? weight[c] : 1.0f;
    float shift = bias ? bias[c] : 0.0f;
    
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = temp_storage[i];
        val = (val - mean_sh) * invstd_sh;
        y_ptr[i] = val * scale + shift;
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
    
    auto y = torch::empty_like(x);
    
    int threads_x = 256;
    int threads_y = 1;
    int blocks = (N * C + threads_y - 1) / threads_y;
    int shared_mem_size = H * W * sizeof(float);
    
    instance_norm_kernel<<<blocks, dim3(threads_x, threads_y), shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA)");
}

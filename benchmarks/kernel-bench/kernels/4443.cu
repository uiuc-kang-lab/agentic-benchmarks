#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ float c_weight[4096];
__constant__ float c_bias[4096];

__inline__ __device__ float warpReduceSum(float val) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xffffffff, val, offset);
#else
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down(val, offset);
#endif
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

    val = (threadIdx.x < blockDim.x/warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void instance_norm_kernel_concurrent(
    const float* __restrict__ x,
    float* __restrict__ y,
    int total_instances,
    int instance_offset,
    int C,
    int HW,
    float eps) 
{
    int instance_id = instance_offset + blockIdx.x * 2 + threadIdx.y;
    if (instance_id >= total_instances) return;

    int n = instance_id / C;
    int c = instance_id % C;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;
    
    float sum = 0, sq_sum = 0;
    const int vec_size = 4;
    int idx = threadIdx.x * vec_size;

    while (idx < HW) {
        float4 vals = *reinterpret_cast<const float4*>(x_ptr + idx);
        sum += vals.x + vals.y + vals.z + vals.w;
        sq_sum += vals.x*vals.x + vals.y*vals.y + vals.z*vals.z + vals.w*vals.w;
        idx += blockDim.x * vec_size * 2;
    }

    float reduce_sum = blockReduceSum(sum);
    float reduce_sq = blockReduceSum(sq_sum);

    __shared__ float mean_shared, var_shared;
    if (threadIdx.x == 0) {
        float mean = reduce_sum / HW;
        float var = fmaxf(reduce_sq / HW - (mean * mean), 0.0f);
        mean_shared = mean;
        var_shared = var;
    }
    __syncthreads();

    const float scale = c_weight[c];
    const float bias = c_bias[c];
    const float inv_std = rsqrtf(var_shared + eps);

    idx = threadIdx.x * vec_size;
    while (idx < HW) {
        float4 vals = *reinterpret_cast<const float4*>(x_ptr + idx);
        vals.x = (vals.x - mean_shared) * inv_std * scale + bias;
        vals.y = (vals.y - mean_shared) * inv_std * scale + bias;
        vals.z = (vals.z - mean_shared) * inv_std * scale + bias;
        vals.w = (vals.w - mean_shared) * inv_std * scale + bias;
        *reinterpret_cast<float4*>(y_ptr + idx) = vals;
        idx += blockDim.x * vec_size * 2;
    }
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) 
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D (N,C,H,W)");

    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    const int HW = H*W;
    const int total_instances = N * C;
    TORCH_CHECK(C <= 4096, "Channels exceed constant memory limit");

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    cudaMemcpyToSymbolAsync(c_weight, weight.data_ptr<float>(), C*sizeof(float), 0, cudaMemcpyDeviceToDevice, stream1);
    cudaMemcpyToSymbolAsync(c_bias, bias.data_ptr<float>(), C*sizeof(float), 0, cudaMemcpyDeviceToDevice, stream2);

    auto y = torch::empty_like(x);
    const int threads = 256;
    const int instances_per_block = 2;
    
    dim3 blocks((total_instances + instances_per_block-1) / instances_per_block);
    dim3 threads_dim(threads/2, 2);

    instance_norm_kernel_concurrent<<<blocks, threads_dim, 0, stream1>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        total_instances,
        0,
        C,
        HW,
        static_cast<float>(eps)
    );

    instance_norm_kernel_concurrent<<<blocks, threads_dim, 0, stream2>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        total_instances,
        1,
        C,
        HW,
        static_cast<float>(eps)
    );

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm optimized with stream concurrency");
}

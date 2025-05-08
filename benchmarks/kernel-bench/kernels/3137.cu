#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

template <int VEC_SIZE, bool ALIGNED>
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int vec_tid = tid * VEC_SIZE;
    const int vec_stride = blockDim.x * VEC_SIZE;

    __shared__ float sdata[(THREADS_PER_BLOCK + 31) / 32];
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Vectorized max reduction
    float max_val = -INFINITY;
    if (ALIGNED) {
        #pragma unroll
        for (int i = vec_tid; i < num_features; i += vec_stride) {
            float4 vec = *reinterpret_cast<const float4*>(x_row + i);
            max_val = fmaxf(max_val, fmaxf(fmaxf(vec.x, vec.y), fmaxf(vec.z, vec.w)));
        }
    } else {
        for (int i = tid; i < num_features; i += blockDim.x) {
            max_val = fmaxf(max_val, x_row[i]);
        }
    }

    // Warp-level max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    if (tid % 32 == 0)
        sdata[tid / 32] = max_val;
    __syncthreads(); // Ensure maximum is written to shared memory

    // Final block max reduction
    if (tid < 32) {
        max_val = tid < (blockDim.x / 32) ? sdata[tid] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        if (tid == 0)
            sdata[0] = max_val;
    }
    __syncthreads(); // Ensure final max is available for all threads
    max_val = sdata[0];

    // Vectorized sum reduction
    float sum_val = 0.0f;
    if (ALIGNED) {
        #pragma unroll
        for (int i = vec_tid; i < num_features; i += vec_stride) {
            float4 vec = *reinterpret_cast<const float4*>(x_row + i);
            float4 exp_vec = {__expf(vec.x - max_val), __expf(vec.y - max_val),
                             __expf(vec.z - max_val), __expf(vec.w - max_val)};
            *reinterpret_cast<float4*>(y_row + i) = exp_vec;
            sum_val += exp_vec.x + exp_vec.y + exp_vec.z + exp_vec.w;
        }
    } else {
        for (int i = tid; i < num_features; i += blockDim.x) {
            float exp_val = __expf(x_row[i] - max_val);
            y_row[i] = exp_val;
            sum_val += exp_val;
        }
    }

    // Warp-level sum reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);

    if (tid % 32 == 0)
        sdata[tid / 32] = sum_val;
    __syncthreads(); // Ensure sums are written to shared memory

    // Final block sum reduction
    if (tid < 32) {
        sum_val = tid < (blockDim.x / 32) ? sdata[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        if (tid == 0)
            sdata[0] = sum_val;
    }
    __syncthreads(); // Ensure normalized factor is available for all

    // Vectorized normalization
    const float inv_sum = 1.0f / sdata[0];
    if (ALIGNED) {
        #pragma unroll
        for (int i = vec_tid; i < num_features; i += vec_stride) {
            float4 vec = *reinterpret_cast<float4*>(y_row + i);
            vec.x *= inv_sum;
            vec.y *= inv_sum;
            vec.z *= inv_sum;
            vec.w *= inv_sum;
            *reinterpret_cast<float4*>(y_row + i) = vec;
        }
    } else {
        for (int i = tid; i < num_features; i += blockDim.x)
            y_row[i] *= inv_sum;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    const int vec_size = 4;
    const bool aligned = (num_features % vec_size) == 0;
    const int shared_mem = ((THREADS_PER_BLOCK + 31) / 32) * sizeof(float);

    if (aligned) {
        softmax_kernel<vec_size, true><<<batch_size, THREADS_PER_BLOCK, shared_mem>>>(x, y, num_features);
    } else {
        softmax_kernel<1, false><<<batch_size, THREADS_PER_BLOCK, shared_mem>>>(x, y, num_features);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

template <int VEC_SIZE>
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    constexpr int ELEMENTS_PER_LOAD = VEC_SIZE == 1 ? 1 : 4;
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    extern __shared__ float smem[];
    float* warp_maxs = smem;
    float* warp_sums = smem + (THREADS_PER_BLOCK / 32);

    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;

    // Vectorized max reduction --------------
    float max_val = -INFINITY;
    for (int i = tid * ELEMENTS_PER_LOAD; i < num_features; i += THREADS_PER_BLOCK * ELEMENTS_PER_LOAD) {
        if (VEC_SIZE == 4) {
            float4 vals = *reinterpret_cast<const float4*>(x_row + i);
            max_val = fmaxf(max_val, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
        } else {
            for (int j = 0; j < ELEMENTS_PER_LOAD && (i + j) < num_features; ++j) {
                max_val = fmaxf(max_val, __ldg(x_row + i + j));
            }
        }
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    
    if (lane_id == 0)
        warp_maxs[warp_id] = max_val;
    __syncthreads();

    // Block reduce max
    // Block reduce max using a single thread
    if (threadIdx.x == 0) {
        float final_max = warp_maxs[0];
        int num_warps = blockDim.x / 32;
        for (int i = 1; i < num_warps; i++) {
            final_max = fmaxf(final_max, warp_maxs[i]);
        }
        warp_maxs[0] = final_max;
    }
    __syncthreads();
    const float row_max = warp_maxs[0];

    // Vectorized sum calculation ------------
    float sum_val = 0.0f;
    for (int i = tid * ELEMENTS_PER_LOAD; i < num_features; i += THREADS_PER_BLOCK * ELEMENTS_PER_LOAD) {
        if (VEC_SIZE == 4) {
            float4 vals = *reinterpret_cast<const float4*>(x_row + i);
            vals.x = __expf(vals.x - row_max);
            vals.y = __expf(vals.y - row_max);
            vals.z = __expf(vals.z - row_max);
            vals.w = __expf(vals.w - row_max);
            *reinterpret_cast<float4*>(y_row + i) = vals;
            sum_val += vals.x + vals.y + vals.z + vals.w;
        } else {
            for (int j = 0; j < ELEMENTS_PER_LOAD && (i + j) < num_features; ++j) {
                const float val = __expf(__ldg(x_row + i + j) - row_max);
                y_row[i + j] = val;
                sum_val += val;
            }
        }
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
    
    if (lane_id == 0)
        warp_sums[warp_id] = sum_val;
    __syncthreads();

    // Block reduce sum
    if (tid < 32) {
        sum_val = tid < (blockDim.x / 32) ? warp_sums[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
    }
    __syncthreads();
    const float row_sum = sum_val;

    // Vectorized normalization --------------
    for (int i = tid * ELEMENTS_PER_LOAD; i < num_features; i += THREADS_PER_BLOCK * ELEMENTS_PER_LOAD) {
        if (VEC_SIZE == 4) {
            float4 vals = *reinterpret_cast<float4*>(y_row + i);
            vals.x /= row_sum;
            vals.y /= row_sum;
            vals.z /= row_sum;
            vals.w /= row_sum;
            *reinterpret_cast<float4*>(y_row + i) = vals;
        } else {
            for (int j = 0; j < ELEMENTS_PER_LOAD && (i + j) < num_features; ++j) {
                y_row[i + j] /= row_sum;
            }
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    const int vec_size = (num_features % 4 == 0) ? 4 : 1;
    const dim3 grid_dim(batch_size);
    const int smem = (THREADS_PER_BLOCK / 32) * sizeof(float) * 2;

    if (vec_size == 4) {
        softmax_kernel<4><<<grid_dim, THREADS_PER_BLOCK, smem>>>(x, y, num_features);
    } else {
        softmax_kernel<1><<<grid_dim, THREADS_PER_BLOCK, smem>>>(x, y, num_features);
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kFloat32,
              "Need CUDA 2D float tensor");

    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized softmax forward (CUDA)");
}

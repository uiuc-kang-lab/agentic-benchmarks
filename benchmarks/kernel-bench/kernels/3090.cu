#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32

template <int BLOCK_SIZE>
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <int BLOCK_SIZE>
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void softmax_hybrid_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int batch_idx = blockIdx.x;
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    extern __shared__ float shared_data[];
    float* warp_maxes = shared_data;
    float* warp_sums = shared_data + num_warps;

    float thread_max = -FLT_MAX;
    
    if (num_features % 4 == 0 && (size_t)&x_row[tid] % 16 == 0) {
        float4* x_row_vec = (float4*)x_row;
        for (int i = tid; i < num_features/4; i += BLOCK_SIZE) {
            float4 vals = x_row_vec[i];
            thread_max = max(thread_max, max(max(vals.x, vals.y), max(vals.z, vals.w)));
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < num_features; i += BLOCK_SIZE) {
            thread_max = max(thread_max, x_row[i]);
        }
    }

    float warp_max = warp_reduce_max<BLOCK_SIZE>(thread_max);
    
    if (lane_id == 0) {
        warp_maxes[warp_id] = warp_max;
    }
    __syncthreads();

    if (warp_id == 0 && tid < num_warps) {
        float val = warp_maxes[tid];
        val = warp_reduce_max<BLOCK_SIZE>(val);
        if (tid == 0) {
            warp_maxes[0] = val;
        }
    }
    __syncthreads();

    const float row_max = warp_maxes[0];
    float thread_sum = 0.0f;

    if (num_features % 4 == 0 && (size_t)&y_row[tid] % 16 == 0) {
        float4* x_row_vec = (float4*)x_row;
        float4* y_row_vec = (float4*)y_row;
        for (int i = tid; i < num_features/4; i += BLOCK_SIZE) {
            float4 vals = x_row_vec[i];
            float4 exp_vals;
            exp_vals.x = __expf(vals.x - row_max);
            exp_vals.y = __expf(vals.y - row_max);
            exp_vals.z = __expf(vals.z - row_max);
            exp_vals.w = __expf(vals.w - row_max);
            y_row_vec[i] = exp_vals;
            thread_sum += exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < num_features; i += BLOCK_SIZE) {
            float exp_val = __expf(x_row[i] - row_max);
            y_row[i] = exp_val;
            thread_sum += exp_val;
        }
    }

    float warp_sum = warp_reduce_sum<BLOCK_SIZE>(thread_sum);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0 && tid < num_warps) {
        float val = warp_sums[tid];
        val = warp_reduce_sum<BLOCK_SIZE>(val);
        if (tid == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    const float inv_sum = __frcp_rn(warp_sums[0]);

    if (num_features % 4 == 0 && (size_t)&y_row[tid] % 16 == 0) {
        float4* y_row_vec = (float4*)y_row;
        for (int i = tid; i < num_features/4; i += BLOCK_SIZE) {
            float4 vals = y_row_vec[i];
            vals.x *= inv_sum;
            vals.y *= inv_sum;
            vals.z *= inv_sum;
            vals.w *= inv_sum;
            y_row_vec[i] = vals;
        }
    } else {
        #pragma unroll 4
        for (int i = tid; i < num_features; i += BLOCK_SIZE) {
            y_row[i] *= inv_sum;
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features, int block_size) {
    dim3 grid_dim(batch_size);
    int shared_mem_size = sizeof(float) * (block_size / WARP_SIZE) * 2;

    switch(block_size) {
        case 256:
            softmax_hybrid_kernel<256><<<grid_dim, block_size, shared_mem_size>>>(x, y, num_features);
            break;
        case 512:
            softmax_hybrid_kernel<512><<<grid_dim, block_size, shared_mem_size>>>(x, y, num_features);
            break;
        default:
            softmax_hybrid_kernel<256><<<grid_dim, 256, shared_mem_size>>>(x, y, num_features);
    }
}

torch::Tensor forward(torch::Tensor x, int block_size = 256) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    auto y = torch::empty_like(x);
    softmax_forward_cuda(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x.size(0),
        x.size(1),
        block_size
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)",
          py::arg("x"), py::arg("block_size") = 256);
}
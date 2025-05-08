#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256

template<bool IsAligned>
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Ensure 128-bit alignment
    const float* __restrict__ x_row = x + batch_idx * ((num_features + 3) & ~3);
    float* __restrict__ y_row = y + batch_idx * ((num_features + 3) & ~3);

    extern __shared__ float sdata[];

    float max_val = -INFINITY;
    if (IsAligned) {
        // Process 4 elements at a time when aligned
        for (int i = tid * 4; i < num_features; i += stride * 4) {
            float4 val4;
            if (i + 3 < num_features) {
                val4 = *reinterpret_cast<const float4*>(x_row + i);
                max_val = max(max_val, max(max(val4.x, val4.y), max(val4.z, val4.w)));
            } else {
                for (int j = 0; j < 4 && i + j < num_features; j++) {
                    float val = __ldg(&x_row[i + j]);
                    max_val = max(max_val, val);
                }
            }
        }
    } else {
        for (int i = tid; i < num_features; i += stride) {
            float val = __ldg(&x_row[i]);
            max_val = max(max_val, val);
        }
    }

    sdata[tid] = max_val;
    __syncthreads();

    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    max_val = sdata[0];
    __syncthreads();

    float sum_val = 0.0f;
    if (IsAligned) {
        for (int i = tid * 4; i < num_features; i += stride * 4) {
            float4 out4;
            if (i + 3 < num_features) {
                float4 val4 = *reinterpret_cast<const float4*>(x_row + i);
                out4.x = __expf(val4.x - max_val);
                out4.y = __expf(val4.y - max_val);
                out4.z = __expf(val4.z - max_val);
                out4.w = __expf(val4.w - max_val);
                *reinterpret_cast<float4*>(y_row + i) = out4;
                sum_val += out4.x + out4.y + out4.z + out4.w;
            } else {
                for (int j = 0; j < 4 && i + j < num_features; j++) {
                    float val = __ldg(&x_row[i + j]);
                    float exp_val = __expf(val - max_val);
                    y_row[i + j] = exp_val;
                    sum_val += exp_val;
                }
            }
        }
    } else {
        for (int i = tid; i < num_features; i += stride) {
            float val = __ldg(&x_row[i]);
            float exp_val = __expf(val - max_val);
            y_row[i] = exp_val;
            sum_val += exp_val;
        }
    }

    sdata[tid] = sum_val;
    __syncthreads();

    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    sum_val = sdata[0];
    __syncthreads();

    if (IsAligned) {
        for (int i = tid * 4; i < num_features; i += stride * 4) {
            if (i + 3 < num_features) {
                float4 val4 = *reinterpret_cast<float4*>(y_row + i);
                val4.x /= sum_val;
                val4.y /= sum_val;
                val4.z /= sum_val;
                val4.w /= sum_val;
                *reinterpret_cast<float4*>(y_row + i) = val4;
            } else {
                for (int j = 0; j < 4 && i + j < num_features; j++) {
                    y_row[i + j] /= sum_val;
                }
            }
        }
    } else {
        for (int i = tid; i < num_features; i += stride) {
            y_row[i] /= sum_val;
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    
    int shared_mem_size = sizeof(float) * THREADS_PER_BLOCK;
    
    if (num_features % 4 == 0) {
        softmax_kernel<true><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
    } else {
        softmax_kernel<false><<<grid_dim, block_dim, shared_mem_size>>>(x, y, num_features);
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
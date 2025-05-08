#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define THREADS_PER_BLOCK 256

// Template kernel: vectorized softmax with alignment
template <bool IsAligned>
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    // Compute padded feature stride (aligned to 4)
    const int feature_stride = (num_features + 3) & ~3;
    const float* x_row = x + batch_idx * feature_stride;
    float* y_row = y + batch_idx * feature_stride;

    extern __shared__ float sdata[];

    // Step 1: Compute maximum value in the row
    float max_val = -INFINITY;
    if (IsAligned) {
        for (int i = tid * 4; i < num_features; i += stride * 4) {
            if (i + 3 < num_features) {
                float4 val4 = *reinterpret_cast<const float4*>(x_row + i);
                float m0 = fmaxf(val4.x, val4.y);
                float m1 = fmaxf(val4.z, val4.w);
                max_val = fmaxf(max_val, fmaxf(m0, m1));
            } else {
                for (int j = 0; j < 4 && (i + j) < num_features; j++) {
                    float temp = __ldg(&x_row[i + j]);
                    max_val = fmaxf(max_val, temp);
                }
            }
        }
    } else {
        for (int i = tid; i < num_features; i += stride) {
            float temp = __ldg(&x_row[i]);
            max_val = fmaxf(max_val, temp);
        }
    }

    sdata[tid] = max_val;
    __syncthreads();

    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Step 2: Compute exponentials and sum them
    float sum_val = 0.0f;
    if (IsAligned) {
        for (int i = tid * 4; i < num_features; i += stride * 4) {
            if(i + 3 < num_features) {
                float4 val4 = *reinterpret_cast<const float4*>(x_row + i);
                float4 out4;
                out4.x = __expf(val4.x - max_val);
                out4.y = __expf(val4.y - max_val);
                out4.z = __expf(val4.z - max_val);
                out4.w = __expf(val4.w - max_val);
                *reinterpret_cast<float4*>(y_row + i) = out4;
                sum_val += (out4.x + out4.y + out4.z + out4.w);
            } else {
                for (int j = 0; j < 4 && (i + j) < num_features; j++) {
                    float temp = __ldg(&x_row[i+j]);
                    float exp_val = __expf(temp - max_val);
                    y_row[i+j] = exp_val;
                    sum_val += exp_val;
                }
            }
        }
    } else {
        for (int i = tid; i < num_features; i += stride) {
            float temp = __ldg(&x_row[i]);
            float exp_val = __expf(temp - max_val);
            y_row[i] = exp_val;
            sum_val += exp_val;
        }
    }

    sdata[tid] = sum_val;
    __syncthreads();

    for (unsigned int s = stride / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    sum_val = sdata[0];
    __syncthreads();

    // Step 3: Normalize the result
    if (IsAligned) {
        for (int i = tid * 4; i < num_features; i += stride * 4) {
            if(i+3 < num_features) {
                float4 val4 = *reinterpret_cast<float4*>(y_row + i);
                val4.x /= sum_val;
                val4.y /= sum_val;
                val4.z /= sum_val;
                val4.w /= sum_val;
                *reinterpret_cast<float4*>(y_row + i) = val4;
            } else {
                for (int j = 0; j < 4 && (i+j) < num_features; j++) {
                    y_row[i+j] /= sum_val;
                }
            }
        }
    } else {
        for (int i = tid; i < num_features; i += stride) {
            y_row[i] /= sum_val;
        }
    }
}

// Host function to launch kernels using CUDA streams for pipelined computation
void softmax_forward_cuda_streams(const float* x, float* y, int batch_size, int num_features) {
    // Calculate padded feature stride (aligned to 4)
    int feature_stride = (num_features + 3) & ~3;
    
    // Use a limited number of streams to overlap kernel execution
    int num_streams = batch_size < 4 ? batch_size : 4;
    int chunk_size = batch_size / num_streams;
    int remainder = batch_size % num_streams;

    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    dim3 block_dim(THREADS_PER_BLOCK);
    int shared_mem = THREADS_PER_BLOCK * sizeof(float);

    int batch_offset = 0;
    for (int i = 0; i < num_streams; i++) {
        int current_chunk = chunk_size + (i < remainder ? 1 : 0);
        if (current_chunk == 0) continue;
        dim3 grid_dim(current_chunk);
        const float* x_offset = x + batch_offset * feature_stride;
        float* y_offset = y + batch_offset * feature_stride;

        // Launch vectorized kernel if features are a multiple of 4.
        if (num_features % 4 == 0) {
            softmax_kernel<true><<<grid_dim, block_dim, shared_mem, streams[i]>>>(x_offset, y_offset, num_features);
        } else {
            softmax_kernel<false><<<grid_dim, block_dim, shared_mem, streams[i]>>>(x_offset, y_offset, num_features);
        }
        batch_offset += current_chunk;
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

// C++ forward function called from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);

    auto y = torch::empty_like(x);
    softmax_forward_cuda_streams(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward with streams (CUDA)");
}

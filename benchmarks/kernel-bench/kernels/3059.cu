#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// This kernel implements softmax using __ldg() for read-only global memory accesses
// and uses 128-bit aligned vectorized load/store (via float4) when possible.

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int num_warps = blockSize / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // Shared memory: first num_warps floats for max reduction, next num_warps for sum reduction
    extern __shared__ float sdata[];
    float* s_max = sdata;              // size = num_warps
    float* s_sum = sdata + num_warps;    // size = num_warps

    int row_offset = batch_idx * num_features;

    // Check if the number of features is a multiple of 4 for vectorized accesses
    if (num_features % 4 == 0) {
        int N_vec = num_features / 4;
        const float4* x_vec = reinterpret_cast<const float4*>(x);
        float4* y_vec = reinterpret_cast<float4*>(y);
        int vec_row_offset = batch_idx * N_vec;

        // First pass: compute the maximum value in the row using vectorized loads
        float local_max = -INFINITY;
        for (int i = tid; i < N_vec; i += blockSize) {
            float4 v = __ldg(x_vec + vec_row_offset + i);
            local_max = fmaxf(local_max, v.x);
            local_max = fmaxf(local_max, v.y);
            local_max = fmaxf(local_max, v.z);
            local_max = fmaxf(local_max, v.w);
        }
        // Warp-level reduction using shuffle
        for (int offset_sh = WARP_SIZE / 2; offset_sh > 0; offset_sh /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset_sh));
        }
        if (lane == 0) {
            s_max[tid / WARP_SIZE] = local_max;
        }
        __syncthreads();

        float block_max = -INFINITY;
        if (tid < num_warps)
            block_max = s_max[tid];
        // Reduce the per-warp maxima
        for (int offset_sh = WARP_SIZE / 2; offset_sh > 0; offset_sh /= 2) {
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset_sh));
        }
        if (tid == 0) s_max[0] = block_max;
        __syncthreads();
        float max_val = s_max[0];

        // Second pass: compute exponentials, accumulate the sum and store the exp-ed values using vectorized stores
        float local_sum = 0.0f;
        for (int i = tid; i < N_vec; i += blockSize) {
            float4 v = __ldg(x_vec + vec_row_offset + i);
            float a = __expf(v.x - max_val);
            float b = __expf(v.y - max_val);
            float c = __expf(v.z - max_val);
            float d = __expf(v.w - max_val);
            local_sum += (a + b + c + d);
            float4 t = make_float4(a, b, c, d);
            y_vec[vec_row_offset + i] = t;
        }
        // Warp-level reduction for the sum
        for (int offset_sh = WARP_SIZE / 2; offset_sh > 0; offset_sh /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset_sh);
        }
        if (lane == 0) {
            s_sum[tid / WARP_SIZE] = local_sum;
        }
        __syncthreads();
        float block_sum = 0.0f;
        if (tid < num_warps) block_sum = s_sum[tid];
        for (int offset_sh = WARP_SIZE / 2; offset_sh > 0; offset_sh /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset_sh);
        }
        if (tid == 0) s_sum[0] = block_sum;
        __syncthreads();
        float sum_val = s_sum[0];

        // Third pass: normalize the exponentials using vectorized loads/stores
        for (int i = tid; i < N_vec; i += blockSize) {
            float4 t = y_vec[vec_row_offset + i];
            t.x /= sum_val;
            t.y /= sum_val;
            t.z /= sum_val;
            t.w /= sum_val;
            y_vec[vec_row_offset + i] = t;
        }
    } else {
        // Scalar version using __ldg() for read-only global loads
        float local_max = -INFINITY;
        for (int i = tid; i < num_features; i += blockSize) {
            float val = __ldg(x + row_offset + i);
            local_max = fmaxf(local_max, val);
        }
        // Warp-level reduction
        for (int offset_sh = WARP_SIZE/2; offset_sh > 0; offset_sh /= 2) {
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset_sh));
        }
        if (lane == 0) s_max[tid / WARP_SIZE] = local_max;
        __syncthreads();

        float block_max = -INFINITY;
        if (tid < num_warps) block_max = s_max[tid];
        for (int offset_sh = WARP_SIZE/2; offset_sh > 0; offset_sh /= 2) {
            block_max = fmaxf(block_max, __shfl_down_sync(0xffffffff, block_max, offset_sh));
        }
        if (tid == 0) s_max[0] = block_max;
        __syncthreads();
        float max_val = s_max[0];

        // Second pass: compute exponentials and accumulate the sum, storing results in global memory
        float local_sum = 0.0f;
        for (int i = tid; i < num_features; i += blockSize) {
            float val = __expf(__ldg(x + row_offset + i) - max_val);
            local_sum += val;
            y[row_offset + i] = val;
        }
        // Warp-level reduction for sum
        for (int offset_sh = WARP_SIZE/2; offset_sh > 0; offset_sh /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset_sh);
        }
        if (lane == 0) s_sum[tid / WARP_SIZE] = local_sum;
        __syncthreads();
        float block_sum = 0.0f;
        if (tid < num_warps) block_sum = s_sum[tid];
        for (int offset_sh = WARP_SIZE/2; offset_sh > 0; offset_sh /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset_sh);
        }
        if (tid == 0) s_sum[0] = block_sum;
        __syncthreads();
        float sum_val = s_sum[0];

        // Third pass: normalize the output values
        for (int i = tid; i < num_features; i += blockSize) {
            y[row_offset + i] /= sum_val;
        }
    }
}


void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    // Shared memory for reductions: 2 floats per warp
    size_t shared_mem_size = 2 * num_warps * sizeof(float);
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in softmax_forward_cuda: %s\n", cudaGetErrorString(err));
    }
}


// C++ interface
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");

    int batch_size = x.size(0);
    int num_features = x.size(1);
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), batch_size, num_features);
    return y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward with __ldg and 128-bit aligned accesses (CUDA)");
}

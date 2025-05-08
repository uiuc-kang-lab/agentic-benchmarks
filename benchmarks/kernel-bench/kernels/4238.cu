#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// This kernel implements BatchNorm forward with asynchronous pipelining of memory loads using cp.async
// to overlap global memory transfers with computation in the reduction phase. Each block processes one
// channel. For each sample (n) the contiguous block of H*W elements is processed in tiles with double buffering.

__global__ void async_pipelined_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W) {

    // Each block handles one channel
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int tsize = blockDim.x; // tile size
    int contiguous_len = H * W;  // each n instance has H*W contiguous elements

    // Dynamic shared memory layout:
    // [0, tsize)              : block reduction buffer for sum
    // [tsize, 2*tsize)        : block reduction buffer for sum of squares
    // [2*tsize, 3*tsize)      : tile buffer 0 for pipelined async copy
    // [3*tsize, 4*tsize)      : tile buffer 1 for pipelined async copy
    extern __shared__ float shared_mem[];
    float* red_sum    = shared_mem;            // size: tsize
    float* red_sum_sq = shared_mem + tsize;      // size: tsize
    float* tile_buf0  = shared_mem + 2 * tsize;    // size: tsize
    float* tile_buf1  = shared_mem + 3 * tsize;    // size: tsize

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Phase 1: Reduction using pipelined asynchronous copy
    // Loop over batches (n dimension). For each n, the data for channel c is contiguous
    for (int n = 0; n < N; n++) {
        // Pointer to the contiguous block of H*W elements for this sample and channel c
        const float* ptr = input + ((n * C + c) * contiguous_len);
        int num_tiles = (contiguous_len + tsize - 1) / tsize;

        // Prefetch the first tile into tile_buf0 if there is at least one tile
        if (num_tiles > 0) {
            int offset = 0;
            if (offset + tid < contiguous_len) {
                const float* src = ptr + offset + tid;
                // Asynchronously copy 4 bytes (one float) from global to shared memory
                asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n"
                              :
                              : "r"(tile_buf0 + tid), "l"(src), "n"(4));
            }
            asm volatile ("cp.async.commit_group;" ::: "memory");
        }

        // Process all tiles for this sample
        for (int j = 0; j < num_tiles; j++) {
            int curr_offset = j * tsize;
            // Pre-fetch next tile (if any) into the alternate buffer
            if (j + 1 < num_tiles) {
                int next_offset = (j + 1) * tsize;
                float* next_buf = (j % 2 == 0) ? tile_buf1 : tile_buf0;
                if (next_offset + tid < contiguous_len) {
                    const float* src_next = ptr + next_offset + tid;
                    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;\n"
                                  :
                                  : "r"(next_buf + tid), "l"(src_next), "n"(4));
                }
                asm volatile ("cp.async.commit_group;" ::: "memory");
            }

            // Wait for the current tile's async copies to complete
            asm volatile ("cp.async.wait_all;" ::: "memory");
            __syncthreads();

            // Determine the buffer holding the current tile
            float* curr_buf = (j % 2 == 0) ? tile_buf0 : tile_buf1;
            float val = 0.0f;
            if (curr_offset + tid < contiguous_len) {
                val = curr_buf[tid];
            }
            local_sum += val;
            local_sum_sq += val * val;

            // Synchronize to ensure all threads have finished reading the current tile
            __syncthreads();
        }
    }

    // Block-level reduction across threads for sum and sumsq
    red_sum[tid] = local_sum;
    red_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red_sum[tid] += red_sum[tid + s];
            red_sum_sq[tid] += red_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float mean = red_sum[0] / (N * contiguous_len);
    float var = (red_sum_sq[0] / (N * contiguous_len)) - (mean * mean);

    // Update running statistics if in training mode
    if (training && tid == 0) {
        running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
        running_var[c]  = (1 - momentum) * running_var[c]  + momentum * var;
        // Store mean and variance in shared memory so all threads can access them
        shared_mem[0] = mean;  // reusing red_sum[0]
        shared_mem[1] = var;   
    }
    __syncthreads();
    mean = shared_mem[0];
    var = shared_mem[1];

    // Phase 2: Normalization
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    for (int n = 0; n < N; n++) {
        int base_idx = ((n * C + c) * contiguous_len);
        for (int i = tid; i < contiguous_len; i += tsize) {
            float x = input[base_idx + i];
            output[base_idx + i] = (x - mean) * inv_std * w_val + b_val;
        }
    }
}


// Host function: launches the asynchronous pipelined BatchNorm kernel

torch::Tensor async_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(running_mean);
    CHECK_CUDA(running_var);

    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;
    // Shared memory size: 4 buffers of size 'threads' each
    size_t shared_mem = 4 * threads * sizeof(float);

    // Launch one block per channel
    async_pipelined_batch_norm_kernel<<<C, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        momentum,
        eps,
        output.data_ptr<float>(),
        N, C, H, W);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &async_forward_cuda, "Asynchronous pipelined BatchNorm forward (CUDA)");
}

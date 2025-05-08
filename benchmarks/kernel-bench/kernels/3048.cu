#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Define block configuration
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

// CUDA kernel: Computes softmax for each row using shared memory and warp-level reductions
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    // Each block processes one row
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Allocate shared memory: first part for warp-level max, second for warp-level sum
    extern __shared__ float shared_mem[];
    float* warp_max = shared_mem;
    float* warp_sum = shared_mem + num_warps;

    // Partition the row among warps
    int items_per_warp = (num_features + num_warps - 1) / num_warps;
    int warp_start = warp_id * items_per_warp;
    int warp_end = min(warp_start + items_per_warp, num_features);

    // Compute local maximum within this warp's portion
    float thread_max = -INFINITY;
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        float val = x[batch_idx * num_features + i];
        thread_max = max(thread_max, val);
    }

    // Warp-level reduction for max using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    if (lane == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Reduce across warps (performed by first warp)
    if (tid == 0) {
        float block_max = warp_max[0];
        for (int i = 1; i < num_warps; i++) {
            block_max = max(block_max, warp_max[i]);
        }
        warp_max[0] = block_max;
    }
    __syncthreads();
    float max_val = warp_max[0];

    // Compute exponentials and local sums
    float thread_sum = 0.0f;
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        float exp_val = __expf(x[batch_idx * num_features + i] - max_val);
        y[batch_idx * num_features + i] = exp_val;
        thread_sum += exp_val;
    }

    // Warp-level reduction to compute sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane == 0) {
        warp_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    // Reduce sums across warps
    if (tid == 0) {
        float block_sum = warp_sum[0];
        for (int i = 1; i < num_warps; i++) {
            block_sum += warp_sum[i];
        }
        warp_sum[0] = block_sum;
    }
    __syncthreads();
    float sum_val = warp_sum[0];

    // Normalize to get the final softmax output
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        y[batch_idx * num_features + i] /= sum_val;
    }
}

// Host function that pipelines computation using CUDA streams to overlap work across batch chunks
void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    // Determine the number of streams to use. Here, we use up to 4 streams or fewer if batch_size is small.
    int num_streams = std::min(batch_size, 4);
    int chunk_size = (batch_size + num_streams - 1) / num_streams;
    std::vector<cudaStream_t> streams(num_streams);

    // Create non-blocking CUDA streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    dim3 block(THREADS_PER_BLOCK);
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    size_t shared_mem_size = 2 * num_warps * sizeof(float);

    // Launch kernel asynchronously on each stream over its batch chunk
    for (int s = 0; s < num_streams; s++) {
        int start = s * chunk_size;
        int current_batch = std::min(chunk_size, batch_size - start);
        if (current_batch > 0) {
            dim3 grid(current_batch);
            // Offset pointers for the current chunk
            const float* x_chunk = x + start * num_features;
            float* y_chunk = y + start * num_features;
            softmax_kernel<<<grid, block, shared_mem_size, streams[s]>>>(x_chunk, y_chunk, num_features);
        }
    }

    // Synchronize and destroy streams
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }
}

// PyTorch binding: Forward function exposed to Python
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
    m.def("forward", &forward, "Softmax forward (CUDA) with stream pipelining");
}

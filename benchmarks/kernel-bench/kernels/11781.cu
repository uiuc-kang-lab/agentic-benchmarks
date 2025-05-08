#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

// Kernel using vectorized loads and CUDA streams for overlapping computation and memory transfers
__global__ void streamed_vectorized_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int total_threads = gridDim.x * blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    // Process elements in groups of 4 using vectorized loads
    int num_vec = n / 4;  // number of complete groups of 4
    for (int i = tid; i < num_vec; i += total_threads) {
        float4 lp = __ldg(reinterpret_cast<const float4*>(log_predictions) + i);
        float4 t  = __ldg(reinterpret_cast<const float4*>(targets) + i);
        sum += expf(lp.x) - t.x * lp.x;
        sum += expf(lp.y) - t.y * lp.y;
        sum += expf(lp.z) - t.z * lp.z;
        sum += expf(lp.w) - t.w * lp.w;
    }

    // Process any remaining elements that don't fit into a group of 4
    int tail_start = num_vec * 4;
    for (int i = tail_start + tid; i < n; i += total_threads) {
        float lp = log_predictions[i];
        float t  = targets[i];
        sum += expf(lp) - t * lp;
    }

    // Warp-level reduction using shuffle instructions
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to reduce the sums from different warps in the block
    extern __shared__ float shared[];
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    // Final reduction performed by the first warp of the block
    if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
        sum = shared[threadIdx.x];
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function that sets up and launches the kernel with CUDA streams
torch::Tensor streamed_vectorized_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    int num_vec = n / 4;
    int blocks = (num_vec + threads - 1) / threads;
    blocks = min(blocks, 256);

    int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    streamed_vectorized_kl_kernel<<<blocks, threads, shared_mem, stream>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &streamed_vectorized_kl_forward, "Streamed Vectorized KL divergence (CUDA)");
}
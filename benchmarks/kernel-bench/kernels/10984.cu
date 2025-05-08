#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE 256
#define N_STREAMS 4

// This kernel computes the partial sum of squared differences for a given chunk.
// It uses a grid-stride loop, warp-level and block-level reductions, and atomicAdd
// to accumulate the partial result into a single double value.

template <typename scalar_t>
__global__ void mse_forward_partial_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t count
) {
    double thread_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < count; i += stride) {
        double diff = static_cast<double>(preds[i]) - static_cast<double>(tgts[i]);
        thread_sum += diff * diff;
    }

    // Warp-level reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Shared memory reduction per block
    __shared__ double shared_sum[BLOCK_SIZE / 32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (threadIdx.x < (BLOCK_SIZE / 32)) {
        double block_sum = shared_sum[threadIdx.x];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum_out, block_sum);
        }
    }
}

// Host function that splits the input into chunks and uses multiple CUDA streams
// to overlap kernel computation with asynchronous memory transfers.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    const int64_t chunk_size = (num_elements + N_STREAMS - 1) / N_STREAMS;

    // Allocate a device tensor to hold partial sums (one per stream)
    auto partial_sums = torch::zeros({N_STREAMS}, predictions.options().dtype(at::kDouble));

    // Allocate host pinned memory to asynchronously retrieve each partial result
    double* host_partial = nullptr;
    cudaHostAlloc((void**)&host_partial, N_STREAMS * sizeof(double), cudaHostAllocDefault);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(N_STREAMS);
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernel on each stream for its corresponding chunk
    for (int i = 0; i < N_STREAMS; i++) {
        int64_t offset = i * chunk_size;
        if (offset >= num_elements) break;
        int64_t count = std::min(chunk_size, num_elements - offset);
        int grid_size = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_partial_kernel", ([&] {
            mse_forward_partial_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
                predictions.data_ptr<scalar_t>() + offset,
                targets.data_ptr<scalar_t>() + offset,
                partial_sums.data_ptr<double>() + i,
                count
            );
        }));
        
        // Asynchronously copy the computed partial sum from device to host
        cudaMemcpyAsync(&host_partial[i],
                        partial_sums.data_ptr<double>() + i,
                        sizeof(double),
                        cudaMemcpyDeviceToHost,
                        streams[i]);
    }

    // Synchronize all streams to ensure kernels and memcopies are complete
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Final reduction on the host
    double total_sum = 0.0;
    for (int i = 0; i < N_STREAMS; i++) {
        total_sum += host_partial[i];
    }
    cudaFreeHost(host_partial);

    double mse = total_sum / static_cast<double>(num_elements);
    auto result = torch::full({1}, mse, predictions.options().dtype(at::kDouble));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (CUDA) using multi-stream pipelining");
}

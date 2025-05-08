#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel that processes a chunk of the input to compute KL divergence partial sum
__global__ void pipeline_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial_result,
    int count) {

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    extern __shared__ float shared_data[]; // shared memory for warp-level reduction

    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop over the chunk
    for (int i = tid; i < count; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += __expf(log_pred) - target * log_pred;
    }

    // Intra-warp reduction using shuffle
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write warp result to shared memory
    if (lane_id == 0) {
        shared_data[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces the block's partial sums
    if (warp_id == 0) {
        float block_sum = (lane_id < warps_per_block) ? shared_data[lane_id] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(partial_result, block_sum);
        }
    }
}

// Host function that pipelines computation and memory transfers using CUDA streams
// It splits the input into chunks, launches the reduction kernel asynchronously on each chunk,
// and overlaps the kernel execution with asynchronous memory copies of partial results.

torch::Tensor pipelined_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    // Ensure inputs are contiguous and on CUDA
    log_predictions = log_predictions.contiguous();
    targets = targets.contiguous();
    TORCH_CHECK(log_predictions.is_cuda(), "log_predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    const int n = log_predictions.numel();

    // Define chunk size for pipelined processing; this value can be tuned
    const int chunk_size = 131072;
    const int num_chunks = (n + chunk_size - 1) / chunk_size;

    // Allocate a device tensor to hold a partial sum for each chunk
    auto options = log_predictions.options();
    auto device_partial = torch::zeros({num_chunks}, options);

    // Allocate pinned host memory for asynchronous copy of partial results
    auto host_partial = torch::empty({num_chunks}, options.pinned_memory(true));

    // Create a fixed number of CUDA streams for overlapping kernel execution and memcopies
    const int NUM_STREAMS = 4;
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Kernel launch configuration
    const int block_size = 256;
    const int shared_mem = (block_size / 32) * sizeof(float);

    const float* log_ptr = log_predictions.data_ptr<float>();
    const float* targ_ptr = targets.data_ptr<float>();
    float* device_partial_ptr = device_partial.data_ptr<float>();

    // Process each chunk asynchronously
    for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        int count = std::min(chunk_size, n - offset);

        // Reset the partial result for this chunk to zero asynchronously
        cudaMemsetAsync(device_partial_ptr + i, 0, sizeof(float), streams[i % NUM_STREAMS]);

        int num_blocks = (count + block_size - 1) / block_size;
        // Launch the kernel on slice [offset, offset+count) in the assigned stream
        pipeline_kl_div_kernel<<<num_blocks, block_size, shared_mem, streams[i % NUM_STREAMS]>>>(
            log_ptr + offset,
            targ_ptr + offset,
            device_partial_ptr + i,
            count);

        // Asynchronously copy the computed partial result from device to pinned host memory
        cudaMemcpyAsync(host_partial.data_ptr<float>() + i,
                        device_partial_ptr + i,
                        sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[i % NUM_STREAMS]);
    }

    // Synchronize all streams to ensure all operations are complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Final reduction on host over all chunk partial results
    float final_sum = 0.0f;
    float* host_partial_ptr = host_partial.data_ptr<float>();
    for (int i = 0; i < num_chunks; i++) {
        final_sum += host_partial_ptr[i];
    }

    // Normalize by the total number of elements to obtain the final KL divergence
    float result = final_sum / static_cast<float>(n);
    
    // Return the result as a tensor (with the same device as the input)
    auto output = torch::full({1}, result, options);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pipelined_kl_div_forward, "Pipelined KLDivLoss with overlapping kernel execution and memory transfers (CUDA)");
}

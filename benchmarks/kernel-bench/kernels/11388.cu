#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>
#include <algorithm>


#define BLOCK_SIZE 256

// Optimized kernel using shared memory for computation
__global__ void cosine_similarity_loss_kernel_optimized(const float* __restrict__ predictions,
                                                         const float* __restrict__ targets,
                                                         float* output,
                                                         int N,
                                                         int D) {
    // Allocate shared memory layout:
    // First 2*D floats for loading predictions and targets
    // Next 3 * numWarps floats for warp-level reduction
    extern __shared__ float shared_mem[];

    int row = blockIdx.x;  // one block per row
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Pointers into shared memory
    float* shared_preds = shared_mem;
    float* shared_targets = shared_preds + D;

    // Load one row of predictions and targets from global memory into shared memory
    if (tid < D) {
        shared_preds[tid] = predictions[row * D + tid];
        shared_targets[tid] = targets[row * D + tid];
    }
    __syncthreads();

    // Compute partial reductions
    float sum_dot = 0.0f;
    float sum_pred_sq = 0.0f;
    float sum_target_sq = 0.0f;

    for (int i = tid; i < D; i += blockSize) {
        float p = shared_preds[i];
        float t = shared_targets[i];
        sum_dot += p * t;
        sum_pred_sq += p * p;
        sum_target_sq += t * t;
    }

    // Warp-level reduction using shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_dot += __shfl_down_sync(mask, sum_dot, offset);
        sum_pred_sq += __shfl_down_sync(mask, sum_pred_sq, offset);
        sum_target_sq += __shfl_down_sync(mask, sum_target_sq, offset);
    }

    int lane = tid & 31;
    int warpId = tid >> 5;
    int numWarps = (blockSize + 31) / 32;

    // Use the remaining shared memory for storing per-warp results
    float* s_dot      = shared_mem + 2 * D;
    float* s_pred_sq  = s_dot + numWarps;
    float* s_target_sq= s_pred_sq + numWarps;

    if (lane == 0) {
        s_dot[warpId] = sum_dot;
        s_pred_sq[warpId] = sum_pred_sq;
        s_target_sq[warpId] = sum_target_sq;
    }
    __syncthreads();

    // Final reduction from warp sums
    if (tid < numWarps) {
        sum_dot      = s_dot[tid];
        sum_pred_sq  = s_pred_sq[tid];
        sum_target_sq= s_target_sq[tid];
        for (int offset = numWarps >> 1; offset > 0; offset /= 2) {
            sum_dot      += __shfl_down_sync(0xffffffff, sum_dot, offset);
            sum_pred_sq  += __shfl_down_sync(0xffffffff, sum_pred_sq, offset);
            sum_target_sq+= __shfl_down_sync(0xffffffff, sum_target_sq, offset);
        }
        if (tid == 0) {
            const float eps = 1e-8f;
            float norm_pred = sqrtf(sum_pred_sq);
            float norm_target = sqrtf(sum_target_sq);
            float denominator = norm_pred * norm_target;
            denominator = fmaxf(denominator, eps);
            float cos_sim = sum_dot / denominator;
            atomicAdd(output, 1.0f - cos_sim);
        }
    }
}

// Host function with CUDA streams to overlap asynchronous memory transfers and kernel execution
// If the input tensors are already on CUDA, the kernel is launched normally.
// Otherwise, the inputs (assumed to be in pinned CPU memory) are split into chunks,
// asynchronously copied to the device, and processed concurrently using multiple streams.

torch::Tensor cosine_similarity_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    auto output = torch::zeros({1}, predictions.options());
    const int block_size = BLOCK_SIZE;
    // Shared memory: 2*D floats for data + 3*numWarps floats for reduction
    int numWarps = (block_size + 31) / 32;
    size_t shmem_size = (2 * D + 3 * numWarps) * sizeof(float);

    // If inputs are already on GPU, launch a single kernel invocation
    if (predictions.is_cuda()) {
        cosine_similarity_loss_kernel_optimized<<<N, block_size, shmem_size>>>(
            predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D
        );
        cudaDeviceSynchronize();
        output.div_(N);
        return output;
    } else {
        // Use CUDA streams to overlap memory transfers (Host -> Device) with kernel execution
        // Assume the CPU tensors are allocated in pinned memory
        int chunk_size = 128;  // Tuneable: number of rows per chunk
        int num_chunks = (N + chunk_size - 1) / chunk_size;
        int num_streams = std::min(num_chunks, 4);
        std::vector<cudaStream_t> streams(num_streams);
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        const float* src_preds = predictions.data_ptr<float>();
        const float* src_targets = targets.data_ptr<float>();

        // Process each chunk asynchronously
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int start = chunk * chunk_size;
            int current_rows = std::min(chunk_size, N - start);
            
            float *d_preds = nullptr, *d_targets = nullptr;
            cudaMalloc(&d_preds, current_rows * D * sizeof(float));
            cudaMalloc(&d_targets, current_rows * D * sizeof(float));
            
            // Asynchronously copy the current chunk from pinned host to device
            cudaMemcpyAsync(d_preds, src_preds + start * D, current_rows * D * sizeof(float), cudaMemcpyHostToDevice, streams[chunk % num_streams]);
            cudaMemcpyAsync(d_targets, src_targets + start * D, current_rows * D * sizeof(float), cudaMemcpyHostToDevice, streams[chunk % num_streams]);
            
            // Launch one block per row in the chunk
            cosine_similarity_loss_kernel_optimized<<<current_rows, block_size, shmem_size, streams[chunk % num_streams]>>>(
                d_preds,
                d_targets,
                output.data_ptr<float>(),
                current_rows,
                D
            );
            
            // Free the temporary device memory asynchronously
            cudaFreeAsync(d_preds, streams[chunk % num_streams]);
            cudaFreeAsync(d_targets, streams[chunk % num_streams]);
        }

        // Synchronize all streams
        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }

        output.div_(N);
        return output;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cosine_similarity_loss_forward, "Cosine Similarity Loss Forward with streaming (CUDA)");
}

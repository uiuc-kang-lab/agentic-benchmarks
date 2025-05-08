#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: Compute per-row cosine similarity loss without using atomic operations.
// Each block processes one row of the input. The result for each row is written to the 'losses' array.

__global__ void cosine_loss_per_row_kernel(const float* __restrict__ predictions,
                                             const float* __restrict__ targets,
                                             float* __restrict__ losses,
                                             int N, int D) {
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // Allocate dynamic shared memory for partial sums. We allocate 3 arrays back-to-back:
    // one for dot products, one for prediction squares, and one for target squares.
    extern __shared__ float sdata[];  // total size: 3 * blockDim.x floats
    float* s_dot      = sdata;
    float* s_pred_sq  = sdata + blockDim.x;
    float* s_target_sq= sdata + 2 * blockDim.x;

    float local_dot = 0.0f;
    float local_pred_sq = 0.0f;
    float local_target_sq = 0.0f;

    // Process elements in the D-dimension in strides
    for (int i = tid; i < D; i += blockDim.x) {
        float p = predictions[row * D + i];
        float t = targets[row * D + i];
        local_dot       += p * t;
        local_pred_sq   += p * p;
        local_target_sq += t * t;
    }

    // Store partial results in shared memory
    s_dot[tid] = local_dot;
    s_pred_sq[tid] = local_pred_sq;
    s_target_sq[tid] = local_target_sq;
    __syncthreads();

    // Reduce the shared memory arrays. Use standard binary reduction.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid]      += s_dot[tid + s];
            s_pred_sq[tid]  += s_pred_sq[tid + s];
            s_target_sq[tid]+= s_target_sq[tid + s];
        }
        __syncthreads();
    }

    // The first thread computes the cosine similarity loss and writes it to the losses array.
    if (tid == 0) {
        const float eps = 1e-8f;
        float dot = s_dot[0];
        float pred_norm = sqrtf(s_pred_sq[0]);
        float target_norm = sqrtf(s_target_sq[0]);
        float denom = pred_norm * target_norm;
        denom = fmaxf(denom, eps);
        float cos_sim = dot / denom;
        // Compute loss for this row
        float row_loss = 1.0f - cos_sim;
        losses[row] = row_loss;
    }
}


// Kernel 2: Reduction kernel to sum an array of floats. This kernel reduces 2 elements per thread.

__global__ void reduce_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < n)
        sum = input[i];
    if (i + blockDim.x < n)
        sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Host function: Two-stage reduction avoiding global atomic operations.
// Stage 1: Launch a kernel with one block per row to compute per-row losses.
// Stage 2: Iteratively reduce the losses array to a single scalar using the reduction kernel.

torch::Tensor two_stage_cosine_loss_forward(torch::Tensor predictions, torch::Tensor targets) {
    // Input validation
    TORCH_CHECK(predictions.dim() == 2, "predictions must be 2D");
    TORCH_CHECK(targets.dim() == 2, "targets must be 2D");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");

    int N = predictions.size(0);
    int D = predictions.size(1);

    // Allocate tensor for per-row losses
    auto losses = torch::empty({N}, predictions.options());
    const int block_size = 256;
    int grid_size = N; // one block per row
    size_t shared_mem = 3 * block_size * sizeof(float);

    cosine_loss_per_row_kernel<<<grid_size, block_size, shared_mem>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        losses.data_ptr<float>(),
        N, D);

    // Stage 2: Reduce the 'losses' array to a single scalar using iterative reduction
    torch::Tensor temp = losses;
    int current_size = N;
    while (current_size > 1) {
        int threads = 256;
        // Each block reduces 2*threads elements
        int blocks = (current_size + threads * 2 - 1) / (threads * 2);
        auto reduced = torch::empty({blocks}, predictions.options());
        size_t shmem_reduce = threads * sizeof(float);
        reduce_kernel<<<blocks, threads, shmem_reduce>>>(
            temp.data_ptr<float>(),
            reduced.data_ptr<float>(),
            current_size);
        current_size = blocks;
        temp = reduced;
    }

    // 'temp' now contains a single element which is the sum of per-row losses
    float total_loss = temp.item<float>();
    // Compute average loss over rows
    float avg_loss = total_loss / static_cast<float>(N);

    // Return the result as a tensor of shape {1}
    return torch::full({1}, avg_loss, predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &two_stage_cosine_loss_forward, "Two-stage Cosine Similarity Loss Forward (CUDA) with minimized atomic operations");
}

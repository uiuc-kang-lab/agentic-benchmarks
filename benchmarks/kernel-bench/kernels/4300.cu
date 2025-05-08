#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Optimized BatchNorm kernel combining warp-level reduction and fused phase computation
__global__ void batch_norm_opt_kernel(
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
    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int warpSize = 32;
    const int numWarps = blockSize / warpSize;
    const unsigned int full_mask = 0xffffffff;

    float sum = 0.0f, sum_sq = 0.0f;

    // Phase 1: Compute partial sums (and squares) using a loop over elements
    // Each thread processes a strided chunk of the data
    // Reorganize loops to improve memory coalescing
    // Process data in W-major order so consecutive threads access consecutive elements
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = tid; w < W; w += blockSize) {
                int idx = ((n * C + c) * H + h) * W + w;
                float val = input[idx];
                sum += val;
                sum_sq += val * val;
            }
        }
    }

    // Intra-warp reduction using warp-level primitives
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(full_mask, sum, offset);
        sum_sq += __shfl_down_sync(full_mask, sum_sq, offset);
    }

    // Allocate shared memory to store per-warp partial results
    // Layout: [0, numWarps-1]: partial sums, [numWarps, 2*numWarps-1]: partial sums of squares
    extern __shared__ float shared_data[];
    if ((tid % warpSize) == 0) {
        int warp_id = tid / warpSize;
        shared_data[warp_id] = sum;
        shared_data[warp_id + numWarps] = sum_sq;
    }
    __syncthreads();

    float total_sum = 0.f, total_sum_sq = 0.f;
    // Final reduction across warps; using a single thread (tid == 0) is fast here since numWarps is small
    if (tid == 0) {
        for (int i = 0; i < numWarps; i++) {
            total_sum += shared_data[i];
            total_sum_sq += shared_data[i + numWarps];
        }

        // Compute mean and variance from reduced sums
        float mean = total_sum / num_elements;
        float var = total_sum_sq / num_elements - mean * mean;

        // Update running statistics if training; else, use stored stats
        if (training) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }

        // Write mean and variance to beginning of shared memory for use in Phase 2
        shared_data[0] = mean;
        shared_data[1] = var;
    }
    __syncthreads();

    // Phase 2: Normalize the input using the computed mean and variance
    float mean = shared_data[0];
    float var = shared_data[1];
    float inv_std = rsqrtf(var + eps);
    float gamma = weight[c];
    float beta = bias[c];

    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - mean) * inv_std * gamma + beta;
    }
}

// Forward CUDA function that wraps the kernel launch

torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    // Validate inputs
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

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;
    int warpSize = 32;
    int numWarps = threads / warpSize;
    // Shared memory: 2 floats per warp (one for partial sum and one for partial sum_sq)
    size_t shared_mem = 2 * numWarps * sizeof(float);

    // Launch one block per channel
    batch_norm_opt_kernel<<<C, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        momentum,
        eps,
        output.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized BatchNorm forward (CUDA)");
}

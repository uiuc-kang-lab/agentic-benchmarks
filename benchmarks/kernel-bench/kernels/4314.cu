#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Optimized kernel using __ldg() and 128-bit (float4) aligned loads/stores
__global__ void batch_norm_kernel_ldg(
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

    const int c = blockIdx.x;
    const int total_elems = N * H * W;  // number of elements per channel
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    // Shared memory for reduction: first blockSize for sum, next blockSize for sum of squares
    extern __shared__ float smem[];
    float* sum_shared = smem;
    float* sum_sq_shared = smem + blockSize;
    
    float mean, var;

    // Pointer offset for current channel
    const float* input_channel = input + c * total_elems;
    
    if (training) {
        // Use vectorized loads if possible
        int vec_count = total_elems / 4; // number of float4 groups
        int remainder = total_elems - vec_count * 4;

        float sum = 0.0f;
        float sum_sq = 0.0f;

        // Process vectorized portion using float4 loads and __ldg()
        const float4* input_vec = reinterpret_cast<const float4*>(input_channel);
        for (int i = tid; i < vec_count; i += blockSize) {
            float4 v = __ldg(&input_vec[i]);
            sum += v.x + v.y + v.z + v.w;
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        
        // Process remaining elements
        for (int i = tid; i < remainder; i += blockSize) {
            int idx = vec_count * 4 + i;
            float val = __ldg(&input_channel[idx]);
            sum += val;
            sum_sq += val * val;
        }
        
        // Store partial sums to shared memory
        sum_shared[tid] = sum;
        sum_sq_shared[tid] = sum_sq;
        __syncthreads();

        // Block reduction for sum and sum of squares
        for (int s = blockSize / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sum_shared[tid] += sum_shared[tid + s];
                sum_sq_shared[tid] += sum_sq_shared[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float total_sum = sum_shared[0];
            float total_sum_sq = sum_sq_shared[0];
            mean = total_sum / total_elems;
            var = (total_sum_sq / total_elems) - (mean * mean);
            
            // Update running statistics
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
            
            // Store computed mean and var in the beginning of shared memory for use in phase 2
            smem[0] = mean;
            smem[1] = var;
        }
        __syncthreads();

        // Broadcast computed mean and var
        mean = smem[0];
        var = smem[1];
    } else {
        // In inference mode, use running stats
        mean = __ldg(&running_mean[c]);
        var = __ldg(&running_var[c]);
    }

    const float inv_std = rsqrtf(var + eps);
    const float w_val = __ldg(&weight[c]);
    const float b_val = __ldg(&bias[c]);

    // Phase 2: Normalize input and write to output using vectorized stores when possible
    float* output_channel = output + c * total_elems;
    int vec_count = total_elems / 4;
    int remainder = total_elems - vec_count * 4;

    const float4* input_vec = reinterpret_cast<const float4*>(input_channel);
    float4* output_vec = reinterpret_cast<float4*>(output_channel);

    for (int i = tid; i < vec_count; i += blockSize) {
        float4 in_val = __ldg(&input_vec[i]);
        float4 out_val;
        out_val.x = (in_val.x - mean) * inv_std * w_val + b_val;
        out_val.y = (in_val.y - mean) * inv_std * w_val + b_val;
        out_val.z = (in_val.z - mean) * inv_std * w_val + b_val;
        out_val.w = (in_val.w - mean) * inv_std * w_val + b_val;
        output_vec[i] = out_val;
    }

    // Process any remaining elements
    for (int i = tid; i < remainder; i += blockSize) {
        int idx = vec_count * 4 + i;
        float val = __ldg(&input_channel[idx]);
        output_channel[idx] = (val - mean) * inv_std * w_val + b_val;
    }
}


// CUDA forward function
torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {
    
    // Input checks
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

    // Using 256 threads per block
    const int threads = 256;
    // Shared memory: allocate 2 floats per thread (for sum and sum_sq)
    const size_t shared_mem = 2 * threads * sizeof(float);

    // Launch one block per channel
    batch_norm_kernel_ldg<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA) with ldg and 128-bit alignment");
}

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Vector type for 4-float aligned loads/stores
typedef float4 vec_t;

__global__ void batch_norm_aligned_kernel(
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
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    // Vector size for aligned loads
    const int vec_size = sizeof(vec_t) / sizeof(float);
    const int aligned_elements = (num_elements / vec_size) * vec_size;

    float my_sum = 0.f;
    float my_sum_sq = 0.f;

    if (training) {
        // Process aligned elements using vector loads
        for (int i = tid * vec_size; i < aligned_elements; i += stride * vec_size) {
            int base_idx = ((i / (H * W)) * C + c) * H * W + (i % (H * W));
            vec_t vec_val = *reinterpret_cast<const vec_t*>(&input[base_idx]);
            
            my_sum += vec_val.x + vec_val.y + vec_val.z + vec_val.w;
            my_sum_sq += vec_val.x * vec_val.x + vec_val.y * vec_val.y + 
                        vec_val.z * vec_val.z + vec_val.w * vec_val.w;
        }

        // Handle remaining elements
        for (int i = aligned_elements + tid; i < num_elements; i += stride) {
            int n = i / (H * W);
            int hw = i % (H * W);
            int idx = ((n * C + c) * H + (hw / W)) * W + (hw % W);
            float val = __ldg(&input[idx]);
            my_sum += val;
            my_sum_sq += val * val;
        }

        // Warp reduction
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
            my_sum_sq += __shfl_down_sync(0xffffffff, my_sum_sq, offset);
        }

        __shared__ float warp_sum[32];
        __shared__ float warp_sum_sq[32];
        int warpId = tid / warpSize;
        int lane = tid % warpSize;

        if (lane == 0) {
            warp_sum[warpId] = my_sum;
            warp_sum_sq[warpId] = my_sum_sq;
        }
        __syncthreads();

        float mean = 0.f, var = 0.f;
        if (tid == 0) {
            float total_sum = 0.f;
            float total_sum_sq = 0.f;
            int num_warps = blockDim.x / warpSize;
            
            for (int i = 0; i < num_warps; ++i) {
                total_sum += warp_sum[i];
                total_sum_sq += warp_sum_sq[i];
            }
            
            mean = total_sum / num_elements;
            var = (total_sum_sq / num_elements) - (mean * mean);
            
            running_mean[c] = (1.f - momentum) * __ldg(&running_mean[c]) + momentum * mean;
            running_var[c] = (1.f - momentum) * __ldg(&running_var[c]) + momentum * var;
            
            warp_sum[0] = mean;
            warp_sum[1] = var;
        }
        __syncthreads();
        
        mean = warp_sum[0];
        var = warp_sum[1];
    } else {
        // Use read-only cache for running stats in inference mode
        mean = __ldg(&running_mean[c]);
        var = __ldg(&running_var[c]);
    }

    const float inv_std = rsqrtf(var + eps);
    const float w_val = __ldg(&weight[c]);
    const float b_val = __ldg(&bias[c]);
    const float scale = w_val * inv_std;
    const float shift = b_val - mean * scale;

    // Process aligned elements using vector operations
    for (int i = tid * vec_size; i < aligned_elements; i += stride * vec_size) {
        int base_idx = ((i / (H * W)) * C + c) * H * W + (i % (H * W));
        vec_t vec_val = *reinterpret_cast<const vec_t*>(&input[base_idx]);
        
        vec_t vec_out;
        vec_out.x = vec_val.x * scale + shift;
        vec_out.y = vec_val.y * scale + shift;
        vec_out.z = vec_val.z * scale + shift;
        vec_out.w = vec_val.w * scale + shift;
        
        *reinterpret_cast<vec_t*>(&output[base_idx]) = vec_out;
    }

    // Handle remaining elements
    for (int i = aligned_elements + tid; i < num_elements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int idx = ((n * C + c) * H + (hw / W)) * W + (hw % W);
        float val = __ldg(&input[idx]);
        output[idx] = val * scale + shift;
    }
}

torch::Tensor forward_cuda(
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

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);
    
    const int threads = 256;
    batch_norm_aligned_kernel<<<C, threads>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward with aligned memory access (CUDA)");
}
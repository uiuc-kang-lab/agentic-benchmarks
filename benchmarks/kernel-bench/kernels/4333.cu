#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__device__ void warpReduce(float& sum, float& sumsq) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }
}

__global__ void batch_norm_balanced_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W,
    const bool training,
    const float momentum,
    const float epsilon) {
    
    const int c = blockIdx.x;
    const int batch_block = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;
    
    const int batches_per_block = (N + gridDim.y - 1) / gridDim.y;
    const int batch_start = batch_block * batches_per_block;
    const int batch_end = min(batch_start + batches_per_block, N);
    
    extern __shared__ float shared[];
    float* warp_sums = shared;
    float* warp_sumsq = &shared[BLOCK_SIZE/WARP_SIZE];
    float* mean_var = &shared[2*(BLOCK_SIZE/WARP_SIZE)];
    
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    
    const int HW = H * W;
    const int CHW = C * HW;
    const int elements_per_thread = (HW + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int n = batch_start; n < batch_end; n++) {
        for (int i = 0; i < elements_per_thread; i++) {
            const int hw_idx = tid + i * BLOCK_SIZE;
            if (hw_idx < HW) {
                const int idx = n * CHW + c * HW + hw_idx;
                const float val = input[idx];
                local_sum += val;
                local_sumsq += val * val;
            }
        }
    }
    
    warpReduce(local_sum, local_sumsq);
    
    if (lane == 0) {
        warp_sums[wid] = local_sum;
        warp_sumsq[wid] = local_sumsq;
    }
    __syncthreads();
    
    if (wid == 0) {
        float sum = (lane < BLOCK_SIZE/WARP_SIZE) ? warp_sums[lane] : 0.0f;
        float sumsq = (lane < BLOCK_SIZE/WARP_SIZE) ? warp_sumsq[lane] : 0.0f;
        warpReduce(sum, sumsq);
        
        if (lane == 0) {
            if (training) {
                atomicAdd(&running_mean[c], momentum * (sum/(batch_end-batch_start)/HW) + (1.0f - momentum) * running_mean[c]);
                atomicAdd(&running_var[c], momentum * (sumsq/(batch_end-batch_start)/HW - sum*sum/(batch_end-batch_start)/(batch_end-batch_start)/HW/HW - running_var[c]));
            }
            mean_var[0] = sum / ((batch_end-batch_start) * HW);
            mean_var[1] = sumsq / ((batch_end-batch_start) * HW) - mean_var[0] * mean_var[0];
        }
    }
    __syncthreads();
    
    const float mean = mean_var[0];
    const float var = mean_var[1];
    const float inv_std = rsqrtf(var + epsilon);
    const float w = weight[c];
    const float b = bias[c];
    
    for (int n = batch_start; n < batch_end; n++) {
        for (int i = 0; i < elements_per_thread; i++) {
            const int hw_idx = tid + i * BLOCK_SIZE;
            if (hw_idx < HW) {
                const int idx = n * CHW + c * HW + hw_idx;
                const float val = input[idx];
                output[idx] = (val - mean) * inv_std * w + b;
            }
        }
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
    
    const int num_batch_blocks = min(32, N);
    dim3 grid(C, num_batch_blocks);
    const int threads = BLOCK_SIZE;
    
    const size_t shared_mem_size = (2 * (BLOCK_SIZE/WARP_SIZE) + 2) * sizeof(float);
    
    batch_norm_balanced_kernel<<<grid, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        training,
        momentum,
        eps
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA)");
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, 
                                const float* __restrict__ targets, 
                                float* __restrict__ output, 
                                const int n) {
    grid_group grid = this_grid();
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    __shared__ float sdata[256];
    const int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and compute initial values using sequential loads to ensure coalescing
    float sum = 0.0f;
    
    #pragma unroll 4
    while(idx < n) {
        // Prefetch next iteration's data
        float pred, targ;
        if (idx + gridDim.x * blockDim.x < n) {
            pred = predictions[idx + gridDim.x * blockDim.x];
            targ = targets[idx + gridDim.x * blockDim.x];
        }
        
        // Compute current iteration
        sum += fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
        idx += gridDim.x * blockDim.x;
    }
    
    // Store to shared memory
    sdata[tid] = sum;
    block.sync();

    // Two-phase reduction: first reduce within warps
    if (warp.meta_group_rank() == 0) {
        // Warp-level reduction using cooperative groups
        float warp_sum = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += warp.shfl_down(warp_sum, offset);
        }
        
        // Write warp results back to shared memory
        if (warp.thread_rank() == 0) {
            sdata[warp.meta_group_rank()] = warp_sum;
        }
    }
    block.sync();

    // Write result for this block to global mem
    if(tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    const int n = predictions.numel();
    const int threads = 256;
    const int blocks = min(65535, (n + threads - 1) / threads);
    
    auto options = predictions.options();
    auto block_results = torch::empty({blocks}, options);
    
    hinge_loss_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        block_results.data_ptr<float>(),
        n
    );

    auto sum = torch::sum(block_results);
    return sum / n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward");
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for computing partial sums per block
__global__ void compute_partial_sums(const float* input, float* block_sums, int numel) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Each thread computes its partial sum
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Reduction within block
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    
    if (tid < 64) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 64];
        float val = vsdata[tid];
        
        for (int offset = 32; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (tid == 0) {
            block_sums[blockIdx.x] = val;
        }
    }
}

// CUDA kernel for final reduction of block sums
__global__ void reduce_block_sums(float* block_sums, float* final_sum, int num_blocks) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    float sum = 0.0f;

    // Each thread reduces multiple block sums if necessary
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_sums[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    
    if (tid < 64) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 64];
        float val = vsdata[tid];
        
        for (int offset = 32; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (tid == 0) {
            *final_sum = val;
        }
    }
}

// CUDA kernel for normalization
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    int numel = input.numel();
    
    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Allocate space for block sums
    auto block_sums = torch::zeros({blocks}, input.options());
    auto final_sum = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* block_sums_ptr = block_sums.data_ptr<float>();
    float* final_sum_ptr = final_sum.data_ptr<float>();

    // First kernel: compute partial sums per block
    compute_partial_sums<<<blocks, threads>>>(input_ptr, block_sums_ptr, numel);
    
    // Second kernel: reduce block sums (single block)
    reduce_block_sums<<<1, threads>>>(block_sums_ptr, final_sum_ptr, blocks);
    
    // Get final norm value
    float norm_val;
    cudaMemcpy(&norm_val, final_sum_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    // Normalize the tensor
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}
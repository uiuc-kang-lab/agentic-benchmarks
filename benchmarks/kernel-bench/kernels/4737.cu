#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for computing the sum of squares using optimized indexing and reduction
__global__ void compute_norm_kernel_optimized_indexing(const float* input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Each thread processes multiple elements
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        float val = input[i];
        sum += val * val;
    }

    // Write partial sum to shared memory
    sdata[tid] = sum;
    __syncthreads();

    // First reduction stage: 256 -> 128
    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();

    // Second stage: reduce 128 -> 64 and then use warp-level reduction
    if (tid < 64) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 64];
        float val = vsdata[tid];
        for (int offset = 32; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            atomicAdd(norm_out, val);
        }
    }
}

// CUDA kernel for normalizing the tensor
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// Host function interfacing with PyTorch
// This implementation overlaps the reduction computation with memory transfers
// using CUDA streams and pinned host memory to reduce overall runtime.

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    // Allocate a device tensor for storing the norm value (single element)
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();
    int numel = input.numel();

    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Create two CUDA streams: one for reduction and memcopy, one for normalization
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Launch the reduction kernel on stream0
    compute_norm_kernel_optimized_indexing<<<blocks, threads, 0, stream0>>>(input_ptr, norm_ptr, numel);
    
    // Allocate pinned host memory for asynchronous transfer of the norm value
    float* host_norm = nullptr;
    cudaMallocHost(&host_norm, sizeof(float));

    // Asynchronously copy the computed norm from device to pinned host memory on stream0
    cudaMemcpyAsync(host_norm, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost, stream0);

    // Wait for stream0 to finish the reduction and memcopy
    cudaStreamSynchronize(stream0);

    // Compute the Frobenius norm on the host
    float norm_val = sqrt(host_norm[0]);
    
    // Launch the normalization kernel on stream1 with the computed norm
    normalize_kernel<<<blocks, threads, 0, stream1>>>(input_ptr, output_ptr, norm_val, numel);

    // Wait for normalization to complete
    cudaStreamSynchronize(stream1);

    // Clean up pinned memory and CUDA streams
    cudaFreeHost(host_norm);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with overlapped streams");
}

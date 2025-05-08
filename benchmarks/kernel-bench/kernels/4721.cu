#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for computing sum of squares with minimized synchronizations
__global__ void compute_norm_kernel_min_sync(const float* input, float* norm_out, int numel) {
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

    // Write partial sum to shared memory
    sdata[tid] = sum;
    __syncthreads(); // Necessary to ensure all partial sums are in shared memory

    // First reduction stage: combine 256 -> 128
    if (tid < 128) {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads(); // Wait for first stage to complete

    // Second stage: reduce 128 -> 64 using shared memory
    if (tid < 64) {
        volatile float* vsdata = sdata; // Volatile to avoid extra syncs
        vsdata[tid] += vsdata[tid + 64];
        float val = vsdata[tid];

        // Warp-level reduction using shfl_down_sync
        for (int offset = 32; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        // Only one thread per block writes the block's result
        if (tid == 0) {
            atomicAdd(norm_out, val);
        }
    }
}

// CUDA kernel for tensor normalization
__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

// Host function interfacing with PyTorch, now using streams
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();
    int numel = input.numel();

    const int threads = 256;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch kernel to compute the sum of squares with minimal synchronizations on stream1
    compute_norm_kernel_min_sync<<<blocks, threads, 0, stream1>>>(input_ptr, norm_ptr, numel);
    
    // Retrieve the computed sum using cudaMemcpyAsync on stream1
    float norm_val;
    cudaMemcpyAsync(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost, stream1);

    // Wait for the first stream to be completed before launching normalization
    cudaStreamSynchronize(stream1);

    // Compute the Frobenius norm
    norm_val = sqrt(norm_val);

    // Launch kernel to normalize the tensor on stream2
    normalize_kernel<<<blocks, threads, 0, stream2>>>(input_ptr, output_ptr, norm_val, numel);

    // Wait for all operations to complete
    cudaStreamSynchronize(stream2);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization with stream overlap");
}
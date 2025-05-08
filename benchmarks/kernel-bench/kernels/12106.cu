#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<unsigned int blockSize>
__global__ void hinge_loss_kernel_optimized(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    float* __restrict__ partial_sums,
    const int n
) {
    __shared__ float sdata[blockSize];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    #pragma unroll
    for (; i < n; i += blockDim.x * gridDim.x) {
        float pred = __ldg(&predictions[i]);
        float targ = __ldg(&targets[i]);
        float loss = fmaxf(0.0f, 1.0f - pred * targ);
        output[i] = loss;
        sum += loss;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    #pragma unroll
    for (unsigned int s = blockSize/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);
    
    const int n = predictions.numel();
    const int threads = 128;
    const int blocks = min((n + threads - 1) / threads, 1024);
    
    torch::Tensor output = torch::empty_like(predictions);
    auto partial_sums = torch::empty({blocks}, predictions.options());
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    hinge_loss_kernel_optimized<256><<<blocks, threads, 0, stream>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        n
    );
    
    auto mean = torch::sum(partial_sums) / n;
    
    cudaStreamDestroy(stream);
    return mean;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Hinge Loss Forward");
}
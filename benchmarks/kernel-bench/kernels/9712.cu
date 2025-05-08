#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// CUDA kernel for depthwise convolution
__global__ void depthwise_conv2d_stream_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int stride,
    int padding,
    int dilation,
    int chunk_start,
    int chunk_size) {

    extern __shared__ float sweight[];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < kernel_h) {
        sweight[tid] = weight[tid];
    }
    __syncthreads();

    // Calculate global position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Process only the assigned chunk of channels
    for (int c = chunk_start; c < chunk_start + chunk_size && c < channels; c++) {
        if (idx < out_w && idy < out_h) {
            float sum = 0.0f;
            
            #pragma unroll
            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = idy * stride - padding + kh * dilation;
                int iw = idx * stride - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ((blockIdx.z * channels + c) * in_h + ih) * in_w + iw;
                    sum += input[in_idx] * sweight[kh];
                }
            }
            
            sum += bias[c];
            int out_idx = ((blockIdx.z * channels + c) * out_h + idy) * out_w + idx;
            output[out_idx] = sum;
        }
    }
}

at::Tensor forward(
    at::Tensor x,
    at::Tensor weight,
    c10::optional<at::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    x = x.contiguous();
    weight = weight.contiguous();

    int batch = x.size(0);
    int channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int kernel_h = weight.size(2);

    if (groups != channels) {
        throw std::invalid_argument("Depthwise convolution requires groups == number of input channels.");
    }

    at::Tensor bias_val;
    if (bias.has_value() && bias.value().defined()) {
        bias_val = bias.value().contiguous();
    } else {
        bias_val = at::zeros({channels}, x.options());
    }

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - 1) / stride + 1;

    auto output = at::empty({batch, channels, out_h, out_w}, x.options());

    // Create CUDA streams
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate chunk size for channel splitting
    int chunk_size = (channels + num_streams - 1) / num_streams;
    
    // Configure kernel launch parameters
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              batch);

    // Shared memory size for kernel weights
    int shmem_size = kernel_h * sizeof(float);

    // Launch kernels in different streams
    for (int i = 0; i < num_streams; i++) {
        int chunk_start = i * chunk_size;
        
        depthwise_conv2d_stream_kernel<<<grid, block, shmem_size, streams[i]>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_val.data_ptr<float>(),
            output.data_ptr<float>(),
            batch,
            channels,
            in_h,
            in_w,
            out_h,
            out_w,
            kernel_h,
            stride,
            padding,
            dilation,
            chunk_start,
            chunk_size
        );
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Depthwise 2D Convolution forward with stream pipelining (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
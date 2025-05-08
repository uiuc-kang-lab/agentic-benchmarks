#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void depthwise_conv2d_coalesced(
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
    int dilation) {
    
    // Use 32x8 thread block for better memory coalescing (warp_size x num_warps)
    const int WARP_SIZE = 32;
    const int lane_id = threadIdx.x;         // Thread index within warp (0-31)
    const int warp_id = threadIdx.y;         // Warp index within block (0-7)
    
    // Calculate batch and channel indices
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;

    // Calculate output coordinates
    // Each warp processes consecutive elements in width dimension
    int ow_start = blockIdx.x * WARP_SIZE;
    int oh = blockIdx.y * blockDim.y + warp_id;
    int ow = ow_start + lane_id;

    // Shared memory for weights with padding to avoid bank conflicts
    extern __shared__ float shared_weights[];
    
    // Load weights into shared memory
    if (warp_id == 0 && lane_id < kernel_h) {
        shared_weights[lane_id] = __ldg(&weight[c * kernel_h + lane_id]);
    }
    __syncthreads();

    // Process only valid output positions
    if (oh < out_h && ow < out_w) {
        float sum = 0.0f;
        
        // Calculate input base indices for coalesced access
        int in_base_h = oh * stride - padding;
        int in_base_w = ow * stride - padding;
        
        // Process convolution with coalesced memory access
        #pragma unroll
        for (int kh = 0; kh < kernel_h; kh++) {
            int ih = in_base_h + kh * dilation;
            int iw = in_base_w;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                // Calculate input index for coalesced access
                int in_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                float in_val = __ldg(&input[in_idx]);
                sum += in_val * shared_weights[kh];
            }
        }
        
        // Add bias
        sum += __ldg(&bias[c]);
        
        // Write output with coalesced access
        int out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
        output[out_idx] = sum;
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

    // Configure block and grid dimensions for coalesced access
    const int WARP_SIZE = 32;
    const int NUM_WARPS = 8;
    dim3 block(WARP_SIZE, NUM_WARPS);  // 32x8 threads
    dim3 grid((out_w + WARP_SIZE - 1) / WARP_SIZE,
              (out_h + NUM_WARPS - 1) / NUM_WARPS,
              batch * channels);

    // Shared memory size with padding to avoid bank conflicts
    int shmem_size = kernel_h * sizeof(float);

    depthwise_conv2d_coalesced<<<grid, block, shmem_size>>>(
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
        dilation
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Memory Access Depthwise 2D Convolution forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = c10::nullopt,
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("groups"));
}
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 4

// Declare constant memory for frequently accessed parameters
__constant__ int const_params[8];  // [k, stride, padding, dilation, in_h, in_w, out_h, out_w]
__constant__ float const_bias[1024];  // Assuming max 1024 channels for bias

template <typename scalar_t>
__global__ void thread_mapped_depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch,
    int channels) {
    
    // Direct 3D thread mapping
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // height
    const int z = blockIdx.z * blockDim.z + threadIdx.z;  // batch*channel
    
    if (x >= const_params[6] || y >= const_params[5]) return;  // out_h, out_w bounds check
    
    const int b = z / channels;
    const int c = z % channels;
    
    if (b >= batch) return;
    
    scalar_t sum = 0;
    
    #pragma unroll
    for (int i = 0; i < const_params[0]; ++i) {  // k
        const int ih = y * const_params[1] - const_params[2] + i * const_params[3];  // stride, padding, dilation
        if (ih >= 0 && ih < const_params[4]) {  // in_h
            #pragma unroll
            for (int j = 0; j < const_params[0]; ++j) {
                const int iw = x * const_params[1] - const_params[2] + j * const_params[3];
                if (iw >= 0 && iw < const_params[5]) {  // in_w
                    const int in_idx = ((b * channels + c) * const_params[4] + ih) * const_params[5] + iw;
                    const int w_idx = (c * const_params[0] + i) * const_params[0] + j;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    const int out_idx = ((b * channels + c) * const_params[6] + y) * const_params[7] + x;
    output[out_idx] = sum + const_bias[c];
}

template <typename scalar_t>
__global__ void thread_mapped_pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    // Direct 3D thread mapping
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // height
    const int z = blockIdx.z * blockDim.z + threadIdx.z;  // batch*out_channel
    
    if (x >= width || y >= height) return;
    
    const int b = z / out_channels;
    const int oc = z % out_channels;
    
    if (b >= batch) return;
    
    scalar_t sum = 0;
    
    // Use shared memory for weight caching
    __shared__ scalar_t shared_weight[BLOCK_SIZE_Z][32];  // Cache weights for current thread block
    
    for (int ic_block = 0; ic_block < in_channels; ic_block += 32) {
        // Collaborative loading of weights into shared memory
        if (threadIdx.x < 32 && ic_block + threadIdx.x < in_channels) {
            shared_weight[threadIdx.z][threadIdx.x] = weight[oc * in_channels + ic_block + threadIdx.x];
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < 32 && ic_block + i < in_channels; ++i) {
            const int in_idx = ((b * in_channels + (ic_block + i)) * height + y) * width + x;
            sum += input[in_idx] * shared_weight[threadIdx.z][i];
        }
        __syncthreads();
    }
    
    const int out_idx = ((b * out_channels + oc) * height + y) * width + x;
    output[out_idx] = sum + const_bias[oc];
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride,
    int padding,
    int dilation) {
    
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    // Copy parameters to constant memory
    int h_params[8] = {k, stride, padding, dilation, in_h, in_w, out_h, out_w};
    cudaMemcpyToSymbol(const_params, h_params, sizeof(int) * 8);
    
    // Copy bias to constant memory if present
    if (depthwise_bias.defined()) {
        cudaMemcpyToSymbol(const_bias, depthwise_bias.data_ptr<float>(), 
                          sizeof(float) * depthwise_bias.numel());
    }
    
    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
    
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
    dim3 blocks(
        (out_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        ((batch * in_channels) + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "thread_mapped_depthwise_conv2d", ([&] {
        thread_mapped_depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_output.data_ptr<scalar_t>(),
            batch,
            in_channels
        );
    }));
    
    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
    
    // Update constant memory for pointwise bias
    if (pointwise_bias.defined()) {
        cudaMemcpyToSymbol(const_bias, pointwise_bias.data_ptr<float>(),
                          sizeof(float) * pointwise_bias.numel());
    }
    
    dim3 blocks_pw(
        (out_w + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (out_h + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
        ((batch * out_channels) + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "thread_mapped_pointwise_conv2d", ([&] {
        thread_mapped_pointwise_conv2d_kernel<scalar_t><<<blocks_pw, threads>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            in_channels,
            out_channels,
            out_h,
            out_w
        );
    }));
    
    return output;
}

at::Tensor toTensor(const py::object& obj) {
    if (obj.is_none()) return at::Tensor();
    try {
        return obj.cast<at::Tensor>();
    } catch (const py::cast_error& e) {
        if (py::hasattr(obj, "data")) {
            return obj.attr("data").cast<at::Tensor>();
        }
        throw std::runtime_error("Expected a torch Tensor or Parameter.");
    }
}

at::Tensor forward_wrapper(
    py::object x_obj,
    py::object depthwise_weight_obj,
    py::object pointwise_weight_obj,
    py::object depthwise_bias_obj,
    py::object pointwise_bias_obj,
    int stride,
    int padding,
    int dilation) {
    
    auto x = toTensor(x_obj);
    auto depthwise_weight = toTensor(depthwise_weight_obj);
    auto pointwise_weight = toTensor(pointwise_weight_obj);
    auto depthwise_bias = toTensor(depthwise_bias_obj);
    auto pointwise_bias = toTensor(pointwise_bias_obj);
    
    return forward_cuda(x, depthwise_weight, pointwise_weight,
                       depthwise_bias, pointwise_bias,
                       stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "3D thread-mapped depthwise separable convolution forward");
}
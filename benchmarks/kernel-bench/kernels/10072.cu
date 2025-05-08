#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define BLOCK_SIZE 16
#define CHANNELS_PER_BLOCK 8

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k, int stride,
    int padding, int dilation) {

    __shared__ scalar_t shared_input[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    const int total_elements = batch * channels * out_h * out_w;
    
    // Stride loop over all elements
    for (int idx = blockIdx.x * total_threads + tid; 
         idx < total_elements; 
         idx += gridDim.x * total_threads) {
        
        const int ow = idx % out_w;
        int temp = idx / out_w;
        const int oh = temp % out_h;
        temp = temp / out_h;
        const int c = temp % channels;
        const int n = temp / channels;

        scalar_t sum = 0;
        
        // Load input tile into shared memory
        const int in_y_start = oh * stride - padding;
        const int in_x_start = ow * stride - padding;
        
        #pragma unroll
        for (int i = 0; i < k; ++i) {
            const int ih = in_y_start + i * dilation;
            if (ih >= 0 && ih < in_h) {
                #pragma unroll
                for (int j = 0; j < k; ++j) {
                    const int iw = in_x_start + j * dilation;
                    if (iw >= 0 && iw < in_w) {
                        const int in_idx = n * channels * in_h * in_w + 
                                         c * in_h * in_w + 
                                         ih * in_w + iw;
                        sum += input[in_idx] * weight[c * k * k + i * k + j];
                    }
                }
            }
        }
        
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[idx] = sum;
    }
}

template <typename scalar_t>
__global__ void pointwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch, int in_channels,
    int out_channels, int h, int w) {

    __shared__ scalar_t shared_input[CHANNELS_PER_BLOCK][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ scalar_t shared_weight[CHANNELS_PER_BLOCK][BLOCK_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int out_x = bx * BLOCK_SIZE + tx;
    const int out_y = by * BLOCK_SIZE + ty;
    const int batch_id = bz / ((out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const int oc_block = bz % ((out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    scalar_t sum = 0;
    
    // Process input channels in blocks
    for (int ic_block = 0; ic_block < in_channels; ic_block += CHANNELS_PER_BLOCK) {
        if (out_y < h && out_x < w) {
            // Cooperatively load input block into shared memory
            for (int c = 0; c < CHANNELS_PER_BLOCK; c++) {
                if (ic_block + c < in_channels) {
                    const int in_idx = batch_id * in_channels * h * w +
                                     (ic_block + c) * h * w +
                                     out_y * w + out_x;
                    shared_input[c][ty * BLOCK_SIZE + tx] = input[in_idx];
                }
            }
        }
        
        // Load weight block into shared memory
        for (int c = 0; c < CHANNELS_PER_BLOCK; c++) {
            if (ic_block + c < in_channels && 
                oc_block * BLOCK_SIZE + ty < out_channels) {
                shared_weight[c][ty] = weight[(oc_block * BLOCK_SIZE + ty) * in_channels + ic_block + c];
            }
        }
        
        __syncthreads();
        
        if (out_y < h && out_x < w) {
            for (int c = 0; c < CHANNELS_PER_BLOCK && ic_block + c < in_channels; c++) {
                sum += shared_input[c][ty * BLOCK_SIZE + tx] * shared_weight[c][ty];
            }
        }
        
        __syncthreads();
    }
    
    if (out_y < h && out_x < w && 
        oc_block * BLOCK_SIZE + ty < out_channels) {
        const int out_idx = batch_id * out_channels * h * w +
                           (oc_block * BLOCK_SIZE + ty) * h * w +
                           out_y * w + out_x;
        if (bias != nullptr) {
            sum += bias[oc_block * BLOCK_SIZE + ty];
        }
        output[out_idx] = sum;
    }
}

torch::Tensor forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& depthwise_weight,
    const torch::Tensor& pointwise_weight,
    const torch::Tensor& depthwise_bias,
    const torch::Tensor& pointwise_bias,
    int stride, int padding, int dilation) {

    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    
    const int batch = x.size(0);
    const int in_channels = x.size(1);
    const int in_h = x.size(2);
    const int in_w = x.size(3);
    const int k = depthwise_weight.size(2);
    const int out_h = (in_h + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    const int out_w = (in_w + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    
    auto depthwise_output = torch::empty({batch, in_channels, out_h, out_w}, x.options());
    
    const int depthwise_threads = BLOCK_SIZE * BLOCK_SIZE;
    const int depthwise_blocks = std::min(65535, (batch * in_channels * out_h * out_w + depthwise_threads - 1) / depthwise_threads);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_cuda", ([&] {
        depthwise_conv2d_kernel<scalar_t><<<depthwise_blocks, dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(
            x.data_ptr<scalar_t>(),
            depthwise_weight.data_ptr<scalar_t>(),
            depthwise_bias.defined() ? depthwise_bias.data_ptr<scalar_t>() : nullptr,
            depthwise_output.data_ptr<scalar_t>(),
            batch, in_channels, in_h, in_w, out_h, out_w,
            k, stride, padding, dilation);
    }));
    
    const int out_channels = pointwise_weight.size(0);
    auto output = torch::empty({batch, out_channels, out_h, out_w}, x.options());
    
    dim3 pointwise_blocks(
        (out_w + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (out_h + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch * ((out_channels + BLOCK_SIZE - 1) / BLOCK_SIZE)
    );
    dim3 pointwise_threads(BLOCK_SIZE, BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "pointwise_conv2d_cuda", ([&] {
        pointwise_conv2d_kernel<scalar_t><<<pointwise_blocks, pointwise_threads>>>(
            depthwise_output.data_ptr<scalar_t>(),
            pointwise_weight.data_ptr<scalar_t>(),
            pointwise_bias.defined() ? pointwise_bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch, in_channels, out_channels, out_h, out_w);
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
    int stride, int padding, int dilation) {
    
    auto x = toTensor(x_obj);
    auto depthwise_weight = toTensor(depthwise_weight_obj);
    auto pointwise_weight = toTensor(pointwise_weight_obj);
    auto depthwise_bias = toTensor(depthwise_bias_obj);
    auto pointwise_bias = toTensor(pointwise_bias_obj);
    
    return forward_cuda(
        x, depthwise_weight, pointwise_weight,
        depthwise_bias, pointwise_bias,
        stride, padding, dilation);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "CUDA depthwise separable convolution forward");
}
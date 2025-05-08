#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void optimized_conv2d_kernel(
    const float4 * __restrict__ input4,
    const float * __restrict__ input,
    const float * __restrict__ weight,
    const float * __restrict__ bias,
    float * __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w,
    int stride,
    int padding) {

    int oc = blockIdx.z % out_channels;
    int n  = blockIdx.z / out_channels;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sh_weight[];
    int filter_elems = in_channels * kernel_size * kernel_size;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    #pragma unroll 4
    for (int idx = tid; idx < filter_elems; idx += blockDim.x * blockDim.y) {
        int ic = idx / (kernel_size * kernel_size);
        int rem = idx % (kernel_size * kernel_size);
        int ki = rem / kernel_size;
        int kj = rem % kernel_size;
        int weight_index = oc * (in_channels * kernel_size * kernel_size) + 
                          ic * (kernel_size * kernel_size) + 
                          ki * kernel_size + kj;
        sh_weight[idx] = weight[weight_index];
    }
    
    __syncthreads();

    if (out_row >= out_h || out_col >= out_w) return;

    int base_in_row = out_row * stride - padding;
    int base_in_col = out_col * stride - padding;
    float sum = 0.0f;

    if (base_in_row >= 0 && base_in_col >= 0 && 
        (base_in_row + kernel_size) <= in_h && 
        (base_in_col + kernel_size) <= in_w) {
        
        for (int ic = 0; ic < in_channels; ic++) {
            int input_channel_offset = n * (in_channels * in_h * in_w) + 
                                     ic * (in_h * in_w);
            #pragma unroll 3
            for (int ki = 0; ki < kernel_size; ki++) {
                int in_row = base_in_row + ki;
                #pragma unroll 3
                for (int kj = 0; kj < kernel_size; kj++) {
                    int in_col = base_in_col + kj;
                    int input_index = input_channel_offset + in_row * in_w + in_col;
                    int filter_index = ic * (kernel_size * kernel_size) + 
                                     ki * kernel_size + kj;
                    sum += input[input_index] * sh_weight[filter_index];
                }
            }
        }
    } else {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int ki = 0; ki < kernel_size; ki++) {
                int in_row = base_in_row + ki;
                if (in_row >= 0 && in_row < in_h) {
                    for (int kj = 0; kj < kernel_size; kj++) {
                        int in_col = base_in_col + kj;
                        if (in_col >= 0 && in_col < in_w) {
                            int input_index = n * (in_channels * in_h * in_w) + 
                                            ic * (in_h * in_w) + 
                                            in_row * in_w + in_col;
                            int filter_index = ic * (kernel_size * kernel_size) + 
                                             ki * kernel_size + kj;
                            sum += input[input_index] * sh_weight[filter_index];
                        }
                    }
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[oc];
    }

    int out_index = n * (out_channels * out_h * out_w) + 
                    oc * (out_h * out_w) + 
                    out_row * out_w + out_col;
    output[out_index] = sum;
}

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
    }

    if (groups != 1 || dilation != 1) {
        if (bias.has_value()) {
            return torch::conv2d(x, weight, bias.value(), 
                               {stride, stride}, 
                               {padding, padding}, 
                               {dilation, dilation}, 
                               groups);
        } else {
            return torch::conv2d(x, weight, torch::Tensor(), 
                               {stride, stride}, 
                               {padding, padding}, 
                               {dilation, dilation}, 
                               groups);
        }
    }

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, 
                             x.options());

    dim3 block(16, 16, 1);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y, batch_size * out_channels); // Adjusted grid size for better coverage
              (out_h + block.y - 1) / block.y,
              batch_size * out_channels);

    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);

    optimized_conv2d_kernel<<<grid, block, shared_mem_size>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_size,
        out_h,
        out_w,
        stride,
        padding
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized CUDA forward function for 2D convolution");
}
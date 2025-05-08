#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int TILE = 16;
const int WARPSIZE = 32;

__global__ void conv2d_warp_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_size,
    int out_h,
    int out_w,
    int padding) {

    __shared__ float sh_weights[1024];
    __shared__ float sh_inputs[256];

    int oc = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;
    int out_row = blockIdx.y * TILE + threadIdx.y;
    int out_col = blockIdx.x * TILE + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    
    for (int c_block = 0; c_block < in_channels; c_block += WARPSIZE) {
        int c_limit = min(WARPSIZE, in_channels - c_block);

        for (int k = tid; k < kernel_size*kernel_size*WARPSIZE; k += blockDim.x*blockDim.y) {
            int c = k % WARPSIZE;
            int k_idx = k / WARPSIZE;
            if (c < c_limit) {
                int ki = k_idx / kernel_size;
                int kj = k_idx % kernel_size;
                sh_weights[k] = weight[oc*in_channels*kernel_size*kernel_size + 
                                     (c_block + c)*kernel_size*kernel_size + ki*kernel_size + kj];
            }
        }

        for (int ki = 0; ki < kernel_size; ++ki) {
            for (int kj = 0; kj < kernel_size; ++kj) {
                int in_y = out_row - padding + ki;
                int in_x = out_col - padding + kj;
                
                if (tid < WARPSIZE && in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    sh_inputs[ki*kernel_size + kj] = input[n*in_channels*in_h*in_w + 
                                                         (c_block + tid)*in_h*in_w + in_y*in_w + in_x];
                }
            }
        }
        __syncthreads();

        if (out_row < out_h && out_col < out_w) {
            for (int c = 0; c < c_limit; ++c) {
                for (int ki = 0; ki < kernel_size; ++ki) {
                    for (int kj = 0; kj < kernel_size; ++kj) {
                        int in_y = out_row - padding + ki;
                        int in_x = out_col - padding + kj;
                        if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                            float w = sh_weights[c*kernel_size*kernel_size + ki*kernel_size + kj];
                            float i_val = sh_inputs[ki*kernel_size + kj];
                            sum += w * i_val;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

        if (out_row < out_h && out_col < out_w) {
        if (bias) sum += bias[oc];
        output[n*out_channels*out_h*out_w + oc*out_h*out_w + out_row*out_w + out_col] = sum;
    }
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
    if (bias.has_value()) CHECK_INPUT(bias.value());

    if (groups != 1 || dilation != 1 || stride != 1)
        return torch::conv2d(x, weight, bias.has_value() ? bias.value() : torch::Tensor(),
                           {stride, stride}, {padding, padding}, {dilation, dilation}, groups);

    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int in_h = x.size(2);
    int in_w = x.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int out_h = (in_h + 2*padding - kernel_size)/1 + 1;
    int out_w = (in_w + 2*padding - kernel_size)/1 + 1;

    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, x.options());

    dim3 grid((out_w + TILE-1)/TILE, (out_h + TILE-1)/TILE, batch_size*out_channels);
    dim3 block(TILE, TILE);

    conv2d_warp_reduce_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr, output.data_ptr<float>(),
        batch_size, in_channels, in_h, in_w, out_channels, kernel_size, out_h, out_w, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2D conv with warp reductions");
}

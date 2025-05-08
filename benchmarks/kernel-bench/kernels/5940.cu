#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__constant__ int constParams[4]; // kernel_size, stride, padding, dilation

template <typename scalar_t>
__device__ __forceinline__ bool in_bounds(int dim, int size) {
    return dim >= 0 && dim < size;
}

template <typename scalar_t>
__device__ __forceinline__ int get_input_index(
    int b, int c, int d, int h, int w,
    int channels, int input_d, int input_h, int input_w) {
    return ((b * channels + c) * input_d + d) * input_h * input_w + h * input_w + w;
}

template <typename scalar_t>
__device__ __forceinline__ void update_max(
    scalar_t &max_val, int &max_index,
    scalar_t val, int input_idx) {
    if (val > max_val) {
        max_val = val;
        max_index = input_idx;
    }
}

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t* indices,
    int batch_size,
    int channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_d * output_h * output_w) return;

    const int w_out = idx % output_w;
    const int h_out = (idx / output_w) % output_h;
    const int d_out = (idx / (output_h * output_w)) % output_d;
    const int c = (idx / (output_d * output_h * output_w)) % channels;
    const int b = idx / (output_d * output_h * output_w * channels);

    const int ksize = constParams[0];
    const int stride = constParams[1];
    const int pad = constParams[2];
    const int dil = constParams[3];

    const int d_start = d_out * stride - pad;
    const int h_start = h_out * stride - pad;
    const int w_start = w_out * stride - pad;

    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    int max_index = -1;

    #pragma unroll
    for (int kd = 0; kd < ksize; ++kd) {
        int d_in = d_start + kd * dil;
        if (!in_bounds<scalar_t>(d_in, input_d)) continue;

        #pragma unroll
        for (int kh = 0; kh < ksize; ++kh) {
            int h_in = h_start + kh * dil;
            if (!in_bounds<scalar_t>(h_in, input_h)) continue;

            #pragma unroll
            for (int kw = 0; kw < ksize; ++kw) {
                int w_in = w_start + kw * dil;
                if (!in_bounds<scalar_t>(w_in, input_w)) continue;

                int input_idx = get_input_index<scalar_t>(b, c, d_in, h_in, w_in,
                    channels, input_d, input_h, input_w);
                scalar_t val = input[input_idx];
                update_max(max_val, max_index, val, input_idx);
            }
        }
    }

    output[idx] = max_val;
    if (indices != nullptr) {
        indices[idx] = max_index;
    }
}

void set_const_params(int ksize, int stride, int pad, int dil) {
    int params[4] = {ksize, stride, pad, dil};
    cudaMemcpyToSymbol(constParams, params, sizeof(int)*4);
}

torch::Tensor max_pool3d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices,
    bool ceil_mode) {
    
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];

    int output_d = ceil_mode ? 
        ceil((input_d + 2*padding - dilation*(kernel_size-1)-1)/float(stride)+1) :
        floor((input_d + 2*padding - dilation*(kernel_size-1)-1)/float(stride)+1);
    int output_h = ceil_mode ?
        ceil((input_h + 2*padding - dilation*(kernel_size-1)-1)/float(stride)+1) :
        floor((input_h + 2*padding - dilation*(kernel_size-1)-1)/float(stride)+1);
    int output_w = ceil_mode ?
        ceil((input_w + 2*padding - dilation*(kernel_size-1)-1)/float(stride)+1) :
        floor((input_w + 2*padding - dilation*(kernel_size-1)-1)/float(stride)+1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ? 
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    set_const_params(kernel_size, stride, padding, dilation);
    
    const int threadsPerBlock = 512;
    int elements = batch_size * channels * output_d * output_h * output_w;
    int blocks = (elements + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward (CUDA)");
}
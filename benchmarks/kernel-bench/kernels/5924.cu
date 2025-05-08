#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define WARP_SIZE 32

template <typename scalar_t>
__device__ void warpReduceMax(scalar_t& val, int& idx) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        scalar_t tmp_val = __shfl_down_sync(0xffffffff, val, offset);
        int tmp_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (tmp_val > val || (tmp_val == val && tmp_idx < idx)) {
            val = tmp_val;
            idx = tmp_idx;
        }
    }
}

template <typename scalar_t>
__global__ void max_pool3d_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    const int batch_size,
    const int channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem[];
    scalar_t* shared_vals = reinterpret_cast<scalar_t*>(shared_mem);
    int* shared_idxs = reinterpret_cast<int*>(shared_vals + blockDim.x);

    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_d * output_h * output_w) return;

    const int w_out = output_idx % output_w;
    const int h_out = (output_idx / output_w) % output_h;
    const int d_out = (output_idx / (output_w * output_h)) % output_d;
    const int c = (output_idx / (output_w * output_h * output_d)) % channels;
    const int b = output_idx / (output_w * output_h * output_d * channels);

    const int d_start = d_out * stride - padding;
    const int h_start = h_out * stride - padding;
    const int w_start = w_out * stride - padding;

    scalar_t thread_max = 0;
    int thread_idx = -1;

    for (int k_d = 0; k_d < kernel_size; k_d++) {
        const int d_in = d_start + k_d * dilation;
        if (d_in < 0 || d_in >= input_d) continue;

        for (int k_h = 0; k_h < kernel_size; k_h++) {
            const int h_in = h_start + k_h * dilation;
            if (h_in < 0 || h_in >= input_h) continue;

            for (int k_w = 0; k_w < kernel_size; k_w++) {
                const int w_in = w_start + k_w * dilation;
                if (w_in < 0 || w_in >= input_w) continue;

                const int input_idx = ((b * channels + c) * input_d + d_in) * input_h * input_w +
                                    h_in * input_w + w_in;
                const scalar_t val = input[input_idx];

                if (val > thread_max) {
                    thread_max = val;
                    thread_idx = input_idx;
                }
            }
        }
    }

    shared_vals[threadIdx.x] = thread_max;
    shared_idxs[threadIdx.x] = thread_idx;
    __syncthreads();

    scalar_t val = thread_max;
    int idx = thread_idx;
    warpReduceMax(val, idx);

    if (threadIdx.x % WARP_SIZE == 0) {
        output[output_idx] = val;
        if (indices != nullptr) {
            indices[output_idx] = idx;
        }
    }
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
    const int batch_size = input_sizes[0];
    const int channels = input_sizes[1];
    const int input_d = input_sizes[2];
    const int input_h = input_sizes[3];
    const int input_w = input_sizes[4];

    const int output_d = ceil_mode ? 
        ceil((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_d + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_h = ceil_mode ?
        ceil((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_h + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);
    const int output_w = ceil_mode ?
        ceil((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1) :
        floor((input_w + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1);

    auto output = torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options());
    auto indices = return_indices ? 
        torch::empty({batch_size, channels, output_d, output_h, output_w}, input.options().dtype(torch::kLong)) :
        torch::Tensor();

    const int threads = 256;
    const int shared_mem_size = 2 * threads * sizeof(float);
    const int blocks = (batch_size * channels * output_d * output_h * output_w + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool3d_forward_cuda", ([&] {
        max_pool3d_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size, channels,
            input_d, input_h, input_w,
            output_d, output_h, output_w,
            kernel_size, stride, padding, dilation);
    }));

    if (return_indices) {
        return torch::stack({output, indices}, 0);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool3d_cuda_forward, "Max Pool 3D forward (CUDA)");
}

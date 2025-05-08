#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Warp-aligned tile dimensions
#define TILE_W 32  // Match warp size
#define TILE_H 4   // Small height for better occupancy

template <typename scalar_t>
__global__ void depthwiseConv2DKernelMinSync(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding) {
    
    // Calculate batch and channel indices
    const int nc = blockIdx.z;
    const int c = nc % in_channels;
    const int n = nc / in_channels;

    // Calculate tile coordinates
    const int tile_start_x = blockIdx.x * TILE_W;
    const int tile_start_y = blockIdx.y * TILE_H;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Shared memory dimensions
    const int shared_width = TILE_W + kernel_size - 1;
    const int shared_height = TILE_H + kernel_size - 1;
    
    extern __shared__ char smem[];
    scalar_t* shmem = reinterpret_cast<scalar_t*>(smem);

    // Warp-aligned loading of input data into shared memory
    // Each warp handles a row of the input tile
    const int lane_id = tx % 32;
    const int warp_id = ty;
    
    // Pre-compute input coordinates for boundary checking
    const int in_start_x = tile_start_x - padding;
    const int in_start_y = tile_start_y - padding;

    // Load input data into shared memory using warp-aligned accesses
    #pragma unroll
    for (int j = warp_id; j < shared_height; j += blockDim.y) {
        const int in_y = in_start_y + j;
        const bool valid_y = (in_y >= 0 && in_y < in_height);
        
        #pragma unroll
        for (int i = lane_id; i < shared_width; i += 32) {
            const int in_x = in_start_x + i;
            const bool valid_x = (in_x >= 0 && in_x < in_width);
            
            scalar_t val = 0;
            if (valid_y && valid_x) {
                const int input_idx = ((n * in_channels + c) * in_height + in_y) * in_width + in_x;
                val = x[input_idx];
            }
            shmem[j * shared_width + i] = val;
        }
    }
    __syncthreads(); // Synchronize once after loading all necessary data

    // Compute output elements
    // Each thread processes one output element
    const int out_x = tile_start_x + tx;
    const int out_y = tile_start_y + ty;
    
    // Pre-compute output bounds check
    const bool valid_output = (out_x < out_width && out_y < out_height);
    
    scalar_t sum = 0;
    if (valid_output) {
        const int sh_x = tx;
        const int sh_y = ty;
        
        // Unrolled convolution computation
        #pragma unroll
        for (int kh = 0; kh < kernel_size; kh++) {
            #pragma unroll
            for (int kw = 0; kw < kernel_size; kw++) {
                const int shared_idx = (sh_y + kh) * shared_width + (sh_x + kw);
                const int weight_idx = (c * kernel_size + kh) * kernel_size + kw;
                sum += shmem[shared_idx] * w[weight_idx];
            }
        }
        sum += b[c];
        const int out_idx = ((n * in_channels + c) * out_height + out_y) * out_width + out_x;
        out[out_idx] = sum;
    }
}

torch::Tensor forward_impl(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int groups) {

    const int batch_size = x.size(0);
    const int in_channels = x.size(1);
    const int in_height = x.size(2);
    const int in_width = x.size(3);
    const int kernel_size = weight.size(2);
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    auto out = torch::empty({batch_size, in_channels, out_height, out_width}, x.options());

    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_width + TILE_W - 1) / TILE_W,
        (out_height + TILE_H - 1) / TILE_H,
        batch_size * in_channels
    );

    const int shared_width = TILE_W + kernel_size - 1;
    const int shared_height = TILE_H + kernel_size - 1;
    const size_t shmem_size = shared_width * shared_height * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "depthwise_conv2d_min_sync", ([&] {
        depthwiseConv2DKernelMinSync<scalar_t><<<grid, block, shmem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            out_height,
            out_width,
            stride,
            padding
        );
    }));

    return out;
}

torch::Tensor forward_wrap(
    torch::Tensor x,
    torch::Tensor weight,
    pybind11::object bias_obj,
    int stride,
    int padding,
    int groups) {
    torch::Tensor bias;
    if (bias_obj.is_none()) {
        bias = torch::zeros({x.size(1)}, x.options());
    } else {
        bias = bias_obj.cast<torch::Tensor>();
    }
    return forward_impl(x, weight, bias, stride, padding, groups);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &forward_wrap,
        "Depthwise conv2d forward with minimal synchronization",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias") = py::none(),
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("groups") = 1
    );
}
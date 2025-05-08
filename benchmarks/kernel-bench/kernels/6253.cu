#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a 3D grid layout to ensure that threads within a warp access contiguous
// output positions along the width dimension, which in turn aligns their input accesses for
// the pooling window. For each output element, we precompute the row base pointer for the
// input, so that the inner loop over the width of the pooling window performs coalesced
// global memory reads.

__global__ void avg_pool3d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    __shared__ int shared_dims[6];  // Store frequently used dimensions
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        shared_dims[0] = in_d;
        shared_dims[1] = in_h;
        shared_dims[2] = in_w;
        shared_dims[3] = out_d;
        shared_dims[4] = out_h;
        shared_dims[5] = out_w;
    }
    __syncthreads();

    // Combine index calculations to reduce register usage
    const int idx = blockIdx.z;
    const int d_out = idx % out_d;
    const int nc_idx = idx / out_d;
    const int c = nc_idx % channels;
    const int n = nc_idx / channels;

    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit to reduce divergent execution
    if (h_out >= shared_dims[4] || w_out >= shared_dims[5]) return;

    // Calculate window boundaries and clamp in one step
    const int d_start = max(d_out * stride - padding, 0);
    const int h_start = max(h_out * stride - padding, 0);
    const int w_start = max(w_out * stride - padding, 0);
    const int d_end = min(d_out * stride - padding + kernel_size, shared_dims[0]);
    const int h_end = min(h_out * stride - padding + kernel_size, shared_dims[1]);
    const int w_end = min(w_out * stride - padding + kernel_size, shared_dims[2]);

    // Precompute base offset for n and c to reduce calculations in loop
    const int nc_offset = (n * channels + c) * in_d;
    float sum = 0.0f;

    #pragma unroll 4
    for (int d = d_start; d < d_end; d++) {
        const int d_offset = (nc_offset + d) * in_h;
        #pragma unroll 4
        for (int h = h_start; h < h_end; h++) {
            // Compute row offset once per iteration
            const int row_offset = (d_offset + h) * in_w;
            const int row_length = w_end - w_start;
            
            #pragma unroll 4
            for (int w = 0; w < row_length; w++) {
                sum += input[row_offset + w_start + w];
            }
        }
    }

    const int pool_volume = kernel_size * kernel_size * kernel_size;
    const int output_idx = ((((n * channels + c) * out_d + d_out) * out_h + h_out) * out_w + w_out);
    output[output_idx] = sum / static_cast<float>(pool_volume);
}

at::Tensor forward(at::Tensor input, int kernel_size, int stride, int padding) {
    // Check that input is a 5D CUDA tensor
    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5-dimensional");
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    int batch_size = input.size(0);
    int channels   = input.size(1);
    int in_d       = input.size(2);
    int in_h       = input.size(3);
    int in_w       = input.size(4);

    // Calculate output dimensions
    int out_d = (in_d + 2 * padding - kernel_size) / stride + 1;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = at::empty({batch_size, channels, out_d, out_h, out_w}, input.options());

    // Define block dimensions to promote coalescing in the width dimension
    dim3 block(32, 8, 1);
    // grid.x and grid.y cover the spatial dimensions (w and h), grid.z covers (n, c, d_out)
    dim3 grid((out_w + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y,
              batch_size * channels * out_d);

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    avg_pool3d_forward_kernel<<<grid, block>>>(
        input_ptr, output_ptr,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_size, stride, padding);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "3D Average Pooling forward (CUDA) with unrolled loops");
}
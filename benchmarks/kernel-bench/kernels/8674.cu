#include <torch/extension.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

// Macros to check tensor properties
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// Constants for shared memory optimization
#define BLOCK_SIZE 256
#define TILE_SIZE 16

using namespace cooperative_groups;

// Custom CUDA kernel for transposed convolution
template<typename scalar_t>
__global__ void conv_transpose3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding) {
    
    // Use cooperative groups for better synchronization
    auto block = this_thread_block();
    
    // Shared memory for input and weight tiles
    extern __shared__ scalar_t shared_mem[];
    scalar_t* shared_input = shared_mem;
    scalar_t* shared_weight = shared_mem + TILE_SIZE * TILE_SIZE;
    
    // Calculate global indices
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_idx = bid * blockDim.x + tid;
    
    // Asynchronous loading of input data to shared memory
    if (tid < TILE_SIZE * TILE_SIZE) {
        const int row = tid / TILE_SIZE;
        const int col = tid % TILE_SIZE;
        if (row < depth && col < channels) {
            shared_input[row * TILE_SIZE + col] = input[row * channels + col];
        }
    }
    
    // Asynchronous loading of weight data to shared memory
    if (tid < TILE_SIZE * TILE_SIZE) {
        const int row = tid / TILE_SIZE;
        const int col = tid % TILE_SIZE;
        if (row < kernel_size && col < kernel_size) {
            shared_weight[row * TILE_SIZE + col] = weight[row * kernel_size + col];
        }
    }
    
    // Ensure all data is loaded
    block.sync();
    
    // Compute output with cooperative groups
    if (global_idx < batch_size * channels * depth * height * width) {
        const int b = global_idx / (channels * depth * height * width);
        const int c = (global_idx / (depth * height * width)) % channels;
        const int d = (global_idx / (height * width)) % depth;
        const int h = (global_idx / width) % height;
        const int w = global_idx % width;
        
        scalar_t sum = 0;
        
        // Compute convolution using shared memory
        #pragma unroll
        for (int kd = 0; kd < kernel_size; kd++) {
            #pragma unroll
            for (int kh = 0; kh < kernel_size; kh++) {
                #pragma unroll
                for (int kw = 0; kw < kernel_size; kw++) {
                    const int id = d * stride - padding + kd;
                    const int ih = h * stride - padding + kh;
                    const int iw = w * stride - padding + kw;
                    
                    if (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        sum += shared_input[id * TILE_SIZE + c] * 
                               shared_weight[kd * TILE_SIZE + kw];
                    }
                }
            }
        }
        
        // Add bias if present
        if (bias != nullptr) {
            sum += bias[c];
        }
        
        output[global_idx] = sum;
    }
}

// Function definition matching the expected parameters
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups) {

    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    if (bias.has_value()) {
        CHECK_INPUT(*bias);
    }

    // Get tensor dimensions
    const auto batch_size = x.size(0);
    const auto channels = x.size(1);
    const auto depth = x.size(2);
    const auto height = x.size(3);
    const auto width = x.size(4);
    const auto kernel_size = weight.size(2);

    // Calculate output dimensions
    const auto output_depth = (depth - 1) * stride[0] - 2 * padding[0] + kernel_size + output_padding[0];
    const auto output_height = (height - 1) * stride[1] - 2 * padding[1] + kernel_size + output_padding[1];
    const auto output_width = (width - 1) * stride[2] - 2 * padding[2] + kernel_size + output_padding[2];

    // Create output tensor
    auto output = torch::zeros({batch_size, channels, output_depth, output_height, output_width},
                             x.options());

    // Calculate grid and block dimensions
    const int total_elements = batch_size * channels * output_depth * output_height * output_width;
    const dim3 block_dim(BLOCK_SIZE);
    const dim3 grid_dim((total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Calculate shared memory size
    const int shared_mem_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    // Launch kernel with dynamic shared memory
    AT_DISPATCH_FLOATING_TYPES(x.type(), "conv_transpose3d_kernel", ([&] {
        conv_transpose3d_kernel<scalar_t><<<grid_dim, block_dim, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            depth,
            height,
            width,
            kernel_size,
            stride[0],
            padding[0],
            output_padding[0]
        );
    }));

    return output;
}

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}
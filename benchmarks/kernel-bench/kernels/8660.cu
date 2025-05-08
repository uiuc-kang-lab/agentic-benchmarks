#include <torch/extension.h>

__device__ void load_input_tile(float* shared_input, const float* input, 
    const int tile_idx, const int tile_size, const int input_size) {
    if (threadIdx.x < tile_size && (tile_idx * tile_size + threadIdx.x) < input_size) {
        shared_input[threadIdx.x] = input[tile_idx * tile_size + threadIdx.x];
    }
}

__device__ void load_weight_tile(float* shared_weight, const float* weight,
    const int tile_idx, const int tile_size, const int weight_size) {
    if (threadIdx.x < tile_size && (tile_idx * tile_size + threadIdx.x) < weight_size) {
        shared_weight[threadIdx.x] = weight[tile_idx * tile_size + threadIdx.x];
    }
}

__device__ void compute_output_point(float* output, const float* shared_input,
    const float* shared_weight, const int3 thread_pos, const int3 dims,
    const int3 stride, const int3 padding) {
    float sum = 0.0f;
    #pragma unroll
    for(int k = 0; k < dims.z; k++) {
        for(int j = 0; j < dims.y; j++) {
            for(int i = 0; i < dims.x; i++) {
                int in_idx = (k * dims.y + j) * dims.x + i;
                int w_idx = (k * dims.y + j) * dims.x + i;
                sum += shared_input[in_idx] * shared_weight[w_idx];
            }
        }
    }
    int out_idx = (thread_pos.z * dims.y + thread_pos.y) * dims.x + thread_pos.x;
    output[out_idx] = sum;
}

__global__ void conv_transpose3d_kernel(
    const float* input, const float* weight, float* output,
    const int3 input_dims, const int3 weight_dims, const int3 output_dims,
    const int3 stride, const int3 padding, const int3 output_padding,
    const int groups) {
    
    extern __shared__ float shared_memory[];
    float* shared_input = shared_memory;
    float* shared_weight = shared_memory + blockDim.x;
    
    const int3 thread_pos = make_int3(blockIdx.x * blockDim.x + threadIdx.x,
                                     blockIdx.y * blockDim.y + threadIdx.y,
                                     blockIdx.z * blockDim.z + threadIdx.z);
    
    if (thread_pos.x >= output_dims.x || thread_pos.y >= output_dims.y ||
        thread_pos.z >= output_dims.z) return;
    
    const int tile_size = 32;
    const int num_tiles = (input_dims.x * input_dims.y * input_dims.z + tile_size - 1) / tile_size;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        load_input_tile(shared_input, input, tile, tile_size,
                       input_dims.x * input_dims.y * input_dims.z);
        load_weight_tile(shared_weight, weight, tile, tile_size,
                        weight_dims.x * weight_dims.y * weight_dims.z);
        __syncthreads();
        
        compute_output_point(output, shared_input, shared_weight,
                           thread_pos, input_dims, stride, padding);
        __syncthreads();
    }
}

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
    if (bias.has_value()) CHECK_INPUT(*bias);
    
    auto input_sizes = x.sizes();
    auto weight_sizes = weight.sizes();
    auto output_sizes = at::conv_transpose3d_output_size(
        input_sizes, weight_sizes, padding, output_padding, stride, groups);
    
    auto output = torch::zeros(output_sizes, x.options());
    
    const dim3 threads(16, 16, 4);
    const dim3 blocks(
        (output_sizes[2] + threads.x - 1) / threads.x,
        (output_sizes[3] + threads.y - 1) / threads.y,
        (output_sizes[4] + threads.z - 1) / threads.z
    );
    
    const int shared_mem_size = 2 * 32 * sizeof(float);
    
    conv_transpose3d_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        make_int3(input_sizes[2], input_sizes[3], input_sizes[4]),
        make_int3(weight_sizes[2], weight_sizes[3], weight_sizes[4]),
        make_int3(output_sizes[2], output_sizes[3], output_sizes[4]),
        make_int3(stride[0], stride[1], stride[2]),
        make_int3(padding[0], padding[1], padding[2]),
        make_int3(output_padding[0], output_padding[1], output_padding[2]),
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Transposed Conv3D forward (CUDA)");
}
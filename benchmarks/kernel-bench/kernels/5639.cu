#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

// Atomic maximum for float and double are implemented using atomicCAS loops

template <typename scalar_t>
__device__ inline scalar_t atomicMaxGeneric(scalar_t* address, scalar_t val);

// Specialization for float
template <>
__device__ inline float atomicMaxGeneric<float>(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(val, old_val);
        int new_int = __float_as_int(new_val);
        old = atomicCAS(address_as_int, assumed, new_int);
    } while (assumed != old);
    return __int_as_float(old);
}

// Specialization for double
template <>
__device__ inline double atomicMaxGeneric<double>(double* address, double val) {
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmax(val, old_val);
        unsigned long long int new_ull = __double_as_longlong(new_val);
        old = atomicCAS(address_as_ull, assumed, new_ull);
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Kernel that partitions each pooling window among multiple blocks per output element
// Each block processes a contiguous slice of the pooling window and then updates the corresponding
// output element via an atomic max. This minimizes global atomic usage to one update per block.

template <typename scalar_t>
__global__ void atomic_maxpool2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int blocksPerOutput
) {
    // Decode grid index to determine which output element and pooling-slice this block handles
    int global_block_id = blockIdx.x; 
    int out_index = global_block_id / blocksPerOutput;  // Unique output element
    int block_offset = global_block_id % blocksPerOutput; // Slice index for this output element

    // Decode b, c, oh, ow from linear out_index
    int ow = out_index % output_width;
    int oh = (out_index / output_width) % output_height;
    int c  = (out_index / (output_width * output_height)) % channels;
    int b  = out_index / (channels * output_height * output_width);

    // Total number of elements in the pooling window
    int pool_area = kernel_size * kernel_size;
    int slice_size = (pool_area + blocksPerOutput - 1) / blocksPerOutput; // Ceiling division
    int slice_start = block_offset * slice_size;
    int slice_end = (slice_start + slice_size < pool_area) ? (slice_start + slice_size) : pool_area;

    // Initialize local maximum to -infinity
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    // Each thread processes a portion of the assigned slice using a grid-stride loop
    for (int i = threadIdx.x; i < (slice_end - slice_start); i += blockDim.x) {
        int pool_idx = slice_start + i;
        int kh = pool_idx / kernel_size;
        int kw = pool_idx % kernel_size;

        int ih = oh * stride - padding + kh * dilation;
        int iw = ow * stride - padding + kw * dilation;

        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            int input_index = b * (channels * input_height * input_width) +
                              c * (input_height * input_width) +
                              ih * input_width + iw;
            scalar_t val = input[input_index];
            local_max = fmax(local_max, val);
        }
    }

    // Reduce local maximum within the block using shared memory
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);
    int tid = threadIdx.x;
    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Thread 0 of the block updates the global output using an atomic max
    if (tid == 0) {
        atomicMaxGeneric(&output[out_index], sdata[0]);
    }
}

// Host function for atomic max pooling
torch::Tensor atomic_maxpool2d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");

    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width  = input.size(3);

    const int output_height = ((input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    const int output_width  = ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    auto output = torch::empty({batch_size, channels, output_height, output_width}, input.options());
    // Initialize output to -infinity based on tensor type
    if (input.scalar_type() == torch::kFloat) {
        output.fill_(-std::numeric_limits<float>::infinity());
    } else {
        output.fill_(-std::numeric_limits<double>::infinity());
    }

    // Determine number of blocks per output element based on pooling area and a fixed thread count
    const int pool_area = kernel_size * kernel_size;
    const int threads = 256;
    int blocksPerOutput = (pool_area + threads - 1) / threads;
    if (blocksPerOutput < 1) blocksPerOutput = 1;

    int total_outputs = batch_size * channels * output_height * output_width;
    int total_blocks = total_outputs * blocksPerOutput;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "atomic_maxpool2d_cuda_forward", ([&] {
        atomic_maxpool2d_kernel<scalar_t><<<total_blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation,
            blocksPerOutput
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &atomic_maxpool2d_cuda_forward, "Max Pool 2D forward with atomic reduction (CUDA)");
}

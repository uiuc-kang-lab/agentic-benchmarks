#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>

// Define block size for kernel launches
#define BLOCK_SIZE 256

// Kernel: performs mean reduction using shared memory on a chunk of data
// Input shape for chunk: [chunk_outer, L, inner] where stride = inner
// Output shape for chunk: [chunk_outer, inner]
template <typename scalar_t>
__global__ void mean_reduce_kernel_shared(const scalar_t* __restrict__ input,
                                           scalar_t* __restrict__ output,
                                           int L,         // reduction dimension length
                                           int stride,    // inner size
                                           int N) {      // total number of output elements in the chunk (chunk_outer * stride)
    int out_idx = blockIdx.x;
    if (out_idx >= N) return;

    // Decode flat output index into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    int base_offset = outer_idx * (L * stride) + inner_idx;

    __shared__ scalar_t sdata[BLOCK_SIZE];
    scalar_t sum = static_cast<scalar_t>(0);

    // Each thread processes elements in a strided loop over the reduction dimension
    for (int i = threadIdx.x; i < L; i += BLOCK_SIZE) {
        sum += __ldg(input + base_offset + i * stride);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0] / static_cast<scalar_t>(L);
    }
}


// Host function: performs mean reduction with overlapping memory transfers and kernel execution using CUDA streams
// Supports both CPU and GPU input. For CPU input, data is processed in chunks with asynchronous memcpy and kernel launches.

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dim
    if (dim < 0) dim += input.dim();

    // Get the input dimensions
    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t L = sizes[dim];

    // Calculate outer_size = product of dims before 'dim'
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    // Calculate inner_size = product of dims after 'dim'
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Total number of output elements = outer_size * inner_size
    int64_t N = outer_size * inner_size;

    // If input is already on GPU, run the kernel directly
    if (input.is_cuda()) {
        auto input_contig = input.contiguous();
        auto output = torch::empty({N}, input.options());
        int blocks = static_cast<int>(N);
        int threads = BLOCK_SIZE;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda_shared", ([&] {
            mean_reduce_kernel_shared<scalar_t><<<blocks, threads>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                static_cast<int>(inner_size),
                static_cast<int>(N)
            );
        }));
        sizes.erase(sizes.begin() + dim);
        output = output.view(sizes);
        return output;
    } else {
        // For CPU input, we overlap host-device transfers with kernel computations using CUDA streams
        // Allocate pinned memory for the final output
        auto output_cpu = torch::empty({N}, input.options().device(torch::kCPU).pinned_memory(true));
        auto input_contig = input.contiguous();

        // Partition along the outer dimension for pipelining
        int64_t total_outer = outer_size;
        int chunk_outer = 64; // You can tune this value
        if (total_outer < chunk_outer) chunk_outer = total_outer;
        int num_chunks = static_cast<int>((total_outer + chunk_outer - 1) / chunk_outer);

        // Create two CUDA streams for double buffering
        cudaStream_t streams[2];
        cudaStreamCreate(&streams[0]);
        cudaStreamCreate(&streams[1]);

        // Element size in bytes
        size_t elem_size = input.element_size();

        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int s = chunk % 2; // Select stream for this chunk
            int64_t start_outer = chunk * chunk_outer;
            int64_t current_chunk = std::min<int64_t>(chunk_outer, total_outer - start_outer);
            
            // Calculate number of elements in this input chunk and output chunk
            size_t input_chunk_elements = current_chunk * L * inner_size;
            size_t output_chunk_elements = current_chunk * inner_size;
            size_t input_bytes = input_chunk_elements * elem_size;
            size_t output_bytes = output_chunk_elements * elem_size;

            // Pointer offset for the input chunk in CPU memory
            char* host_input_ptr = static_cast<char*>(input_contig.data_ptr()) + start_outer * (L * inner_size) * elem_size;

            // Allocate device memory for input chunk
            void* d_input = nullptr;
            cudaMalloc(&d_input, input_bytes);
            // Asynchronously copy input chunk from host to device
            cudaMemcpyAsync(d_input, host_input_ptr, input_bytes, cudaMemcpyHostToDevice, streams[s]);

            // Allocate device memory for output chunk
            void* d_output = nullptr;
            cudaMalloc(&d_output, output_bytes);

            // Launch kernel on this chunk
            int blocks = static_cast<int>(current_chunk * inner_size); // one block per output element in the chunk
            int threads = BLOCK_SIZE;
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda_shared", ([&] {
                mean_reduce_kernel_shared<scalar_t><<<blocks, threads, 0, streams[s]>>>(
                    static_cast<scalar_t*>(d_input),
                    static_cast<scalar_t*>(d_output),
                    static_cast<int>(L),
                    static_cast<int>(inner_size),
                    static_cast<int>(current_chunk * inner_size)
                );
            }));
            
            // Pointer offset for the output chunk in the pinned host output tensor
            char* host_output_ptr = static_cast<char*>(output_cpu.data_ptr()) + start_outer * inner_size * elem_size;
            // Asynchronously copy the output chunk from device to host
            cudaMemcpyAsync(host_output_ptr, d_output, output_bytes, cudaMemcpyDeviceToHost, streams[s]);

            // Synchronize the stream to ensure completion of operations for this chunk
            cudaStreamSynchronize(streams[s]);
            // Free device memory
            cudaFree(d_input);
            cudaFree(d_output);
        }

        // Destroy CUDA streams
        cudaStreamDestroy(streams[0]);
        cudaStreamDestroy(streams[1]);

        // Reshape the output tensor to remove the reduced dimension
        sizes.erase(sizes.begin() + dim);
        auto final_output = output_cpu.view(sizes);
        return final_output;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean Reduction with Overlapped Pipeline using CUDA Streams");
}

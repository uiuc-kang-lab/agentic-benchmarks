#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void hybrid_cross_entropy_loss_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ losses,
    int batch_size,
    int num_classes
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (num_classes <= 32) {
        // For small number of classes, each thread handles one sample
        int gid = bid * blockDim.x + tid;
        int stride = blockDim.x * gridDim.x;
        
        for (int i = gid; i < batch_size; i += stride) {
            const float* logits_i = logits + i * num_classes;
            int target = targets[i];
            if (target < 0 || target >= num_classes) {
                losses[i] = 0.0f;  // Handle invalid targets
                continue;
            }
            
            // Find max logit for numerical stability
            float max_logit = logits_i[0];
            for (int j = 1; j < num_classes; j++) {
                max_logit = fmaxf(max_logit, logits_i[j]);
            }
            
            // Compute sum of exp(logits - max_logit)
            float sum_exp = 0.0f;
            for (int j = 0; j < num_classes; j++) {
                sum_exp += __expf(logits_i[j] - max_logit);
            }
            
            // Compute cross entropy loss
            float log_sum_exp = logf(sum_exp);
            losses[i] = -(logits_i[target] - max_logit - log_sum_exp);
        }
    } else {
        // For large number of classes, use collaborative approach
        float* s_max = sdata;
        float* s_sum = &sdata[blockDim.x];
        
        // Each block processes one sample
        int sample_idx = bid;
        if (sample_idx >= batch_size) return;
        
        const float* sample_logits = logits + sample_idx * num_classes;
        int target = targets[sample_idx];
        
        if (target < 0 || target >= num_classes) {
            if (tid == 0) losses[sample_idx] = 0.0f;
            return;
        }
        
        // Find max logit collaboratively
        float local_max = -FLT_MAX;
        for (int j = tid; j < num_classes; j += blockDim.x) {
            local_max = fmaxf(local_max, sample_logits[j]);
        }
        s_max[tid] = local_max;
        __syncthreads();
        
        // Reduce to find global max
        for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
            }
            __syncthreads();
        }
        float max_val = s_max[0];
        __syncthreads();
        
        // Compute sum of exponentials collaboratively
        float local_sum = 0.0f;
        for (int j = tid; j < num_classes; j += blockDim.x) {
            local_sum += __expf(sample_logits[j] - max_val);
        }
        s_sum[tid] = local_sum;
        __syncthreads();
        
        // Reduce to find total sum
        for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_sum[tid] += s_sum[tid + offset];
            }
            __syncthreads();
        }
        
        // Compute final loss
        if (tid == 0) {
            float log_sum_exp = logf(s_sum[0]);
            losses[sample_idx] = -(sample_logits[target] - max_val - log_sum_exp);
        }
    }
}
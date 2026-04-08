#include <cuda_runtime.h>

// A kernel designed to induce or mitigate shared memory bank conflicts
__global__ void bank_conflict_kernel(float* out) {
    extern __shared__ float s_data[];
    
    int tid = threadIdx.x;
    int padding = 0;
    
    // 32 threads in a warp access elements with specific stride
    // Stride of 32 causes N-way bank conflict, padding alters this.
    int stride = 32 + padding;
    int idx = tid * stride;
    
    // Write to shared memory (potential bank conflict)
    s_data[idx] = (float)tid;
    __syncthreads();
    
    // Read from shared memory (potential bank conflict)
    float val = s_data[idx];
    
    // Avoid optimization
    if (tid == 0) {
        out[0] = val;
    }
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, sizeof(float));
    
    int threadsPerBlock = 256; // 8 warps
    int padding = 0;
    int stride = 32 + padding;
    
    // Allocate enough shared memory for the strided accesses
    int sharedMemSize = threadsPerBlock * stride * sizeof(float);
    
    // Run on a single block to isolate shared memory behavior
    bank_conflict_kernel<<<1, threadsPerBlock, sharedMemSize>>>(d_out);
    cudaDeviceSynchronize();
    
    cudaFree(d_out);
    return 0;
}
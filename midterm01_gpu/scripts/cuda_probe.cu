#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static bool check(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    return false;
}

__global__ void double_values(int* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) values[index] *= 2;
}

int main() {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count < 1) {
        std::fprintf(stderr, "No CUDA device available. Check the host NVIDIA CDI setup.\n");
        return 1;
    }
    cudaDeviceProp properties{};
    if (!check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties")) return 1;
    std::printf("GPU: %s; compute capability: %d.%d; VRAM: %.0f MiB\n",
                properties.name, properties.major, properties.minor,
                static_cast<double>(properties.totalGlobalMem) / (1024 * 1024));

    constexpr int size = 4096;
    std::vector<int> host(size);
    for (int index = 0; index < size; ++index) host[index] = index;
    int* device = nullptr;
    if (!check(cudaMalloc(&device, size * sizeof(int)), "cudaMalloc")) return 1;
    bool ok = check(cudaMemcpy(device, host.data(), size * sizeof(int), cudaMemcpyHostToDevice), "H2D");
    if (ok) {
        double_values<<<(size + 255) / 256, 256>>>(device, size);
        ok = check(cudaGetLastError(), "kernel launch") &&
             check(cudaDeviceSynchronize(), "kernel execution") &&
             check(cudaMemcpy(host.data(), device, size * sizeof(int), cudaMemcpyDeviceToHost), "D2H");
    }
    ok = check(cudaFree(device), "cudaFree") && ok;
    if (!ok) return 1;
    for (int index = 0; index < size; ++index) {
        if (host[index] != index * 2) {
            std::fprintf(stderr, "Incorrect GPU result at index %d\n", index);
            return 1;
        }
    }
    std::printf("PASS: CUDA allocation, H2D, native GPU kernel and D2H verified (%d values).\n", size);
    return 0;
}


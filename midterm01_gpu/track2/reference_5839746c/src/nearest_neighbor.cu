#include "cuda_ndt_backend/nearest_neighbor.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cuda_ndt_backend {
namespace {
void Check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { Check(cudaMalloc(&ptr, bytes), "cudaMalloc nearest-neighbor buffer"); }
    ~DeviceBuffer() noexcept { if (ptr) cudaFree(ptr); }
};
__global__ void NearestKernel(const float* target, int count, const float* queries, int* result) {
    const int query = blockIdx.x, lane = threadIdx.x;
    const float x = queries[3 * query], y = queries[3 * query + 1], z = queries[3 * query + 2];
    float best = CUDART_INF_F;
    int index = -1;
    for (int i = lane; i < count; i += blockDim.x) {
        const float dx = target[3 * i] - x, dy = target[3 * i + 1] - y, dz = target[3 * i + 2] - z;
        const float distance = dx * dx + dy * dy + dz * dz;
        if (distance < best || (distance == best && (index < 0 || i < index))) {
            best = distance;
            index = i;
        }
    }
    __shared__ float distances[256];
    __shared__ int indices[256];
    distances[lane] = best;
    indices[lane] = index;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride) {
            const float other = distances[lane + stride];
            const int other_index = indices[lane + stride];
            if (other < distances[lane] || (other == distances[lane] && other_index >= 0 && (indices[lane] < 0 || other_index < indices[lane]))) {
                distances[lane] = other;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) result[query] = indices[0];
}
}  // namespace

struct NearestNeighbor::Impl {
    int count, capacity;
    DeviceBuffer target, queries, indices;
    Impl(const std::vector<float>& xyz, int max_queries)
        : count(static_cast<int>(xyz.size() / 3)), capacity(max_queries), target(xyz.size() * sizeof(float)),
          queries(static_cast<std::size_t>(max_queries) * 3 * sizeof(float)), indices(static_cast<std::size_t>(max_queries) * sizeof(int)) {
        Check(cudaMemcpy(target.ptr, xyz.data(), xyz.size() * sizeof(float), cudaMemcpyHostToDevice), "upload NN target");
    }
};
NearestNeighbor::NearestNeighbor(const std::vector<float>& target_xyz, int query_capacity) {
    if (target_xyz.empty() || target_xyz.size() % 3 || query_capacity < 1)
        throw std::invalid_argument("Invalid nearest-neighbor target/capacity");
    for (float value : target_xyz) if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite nearest-neighbor target");
    impl_ = std::make_unique<Impl>(target_xyz, query_capacity);
}
NearestNeighbor::~NearestNeighbor() = default;
void NearestNeighbor::Query(const std::vector<float>& query_xyz, std::vector<int>& indices) const {
    const int n = static_cast<int>(query_xyz.size() / 3);
    if (n < 1 || n > impl_->capacity || query_xyz.size() % 3) throw std::invalid_argument("Invalid NN query size");
    for (float value : query_xyz) if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite nearest-neighbor query");
    Check(cudaMemcpy(impl_->queries.ptr, query_xyz.data(), query_xyz.size() * sizeof(float), cudaMemcpyHostToDevice), "upload NN queries");
    NearestKernel<<<n, 256>>>(static_cast<const float*>(impl_->target.ptr), impl_->count,
                             static_cast<const float*>(impl_->queries.ptr), static_cast<int*>(impl_->indices.ptr));
    Check(cudaGetLastError(), "nearest-neighbor kernel launch");
    indices.resize(n);
    // Device-to-host copy synchronizes this kernel; errors terminate the run.
    Check(cudaMemcpy(indices.data(), impl_->indices.ptr, static_cast<std::size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost), "nearest-neighbor kernel completion");
}
}  // namespace cuda_ndt_backend

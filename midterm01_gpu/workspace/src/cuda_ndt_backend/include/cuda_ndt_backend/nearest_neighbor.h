#pragma once
#include <memory>
#include <vector>

namespace cuda_ndt_backend {
// Exact brute-force nearest neighbor, one CUDA block per query. Coordinates
// are local to the current map origin; no Thrust or voxel hash table is used.
class NearestNeighbor {
public:
    NearestNeighbor(const std::vector<float>& target_xyz, int query_capacity);
    ~NearestNeighbor();
    NearestNeighbor(const NearestNeighbor&) = delete;
    NearestNeighbor& operator=(const NearestNeighbor&) = delete;
    void Query(const std::vector<float>& query_xyz, std::vector<int>& indices) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace cuda_ndt_backend

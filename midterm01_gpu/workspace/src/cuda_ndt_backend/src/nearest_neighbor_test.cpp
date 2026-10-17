#include "cuda_ndt_backend/nearest_neighbor.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    try {
        std::vector<float> target, query;
        for (int i = 0; i < 1027; ++i) {
            target.insert(target.end(), {0.03f * i, std::sin(i * .3f), std::cos(i * .17f)});
        }
        for (int i = 0; i < 73; ++i) query.insert(query.end(), {0.4f * i + .013f, -.2f, .4f});
        cuda_ndt_backend::NearestNeighbor nn(target, 73);
        for (int repeat = 0; repeat < 30; ++repeat) {
            std::vector<int> result;
            nn.Query(query, result);
            for (int j = 0; j < 73; ++j) {
                float best = std::numeric_limits<float>::infinity();
                int index = -1;
                for (int i = 0; i < 1027; ++i) {
                    float squared = 0;
                    for (int axis = 0; axis < 3; ++axis) squared += std::pow(target[3 * i + axis] - query[3 * j + axis], 2);
                    if (squared < best) { best = squared; index = i; }
                }
                if (result[j] != index) throw std::runtime_error("CUDA NN differs from CPU exhaustive reference");
            }
        }
        std::cout << "CUDA NN: 2190 query results equal CPU exhaustive reference; uneven target size and repeated launches passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <pcl/common/transforms.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <random>

// Known source->target transform checks the direction and initial-guess contract.
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 1;
    auto source = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    std::mt19937 rng(20261017);
    std::normal_distribution<float> noise(0.0f, 0.2f);
    for (int cluster = 0; cluster < 60; ++cluster) {
        const float x = float(cluster % 5) * 2.7f;
        const float y = float((cluster / 5) % 4) * 3.1f;
        const float z = float(cluster / 20) * 2.2f;
        for (int point = 0; point < 120; ++point)
            source->push_back(pcl::PointXYZ(x + noise(rng), y + noise(rng), z + noise(rng)));
    }
    Eigen::Matrix4f truth = Eigen::Matrix4f::Identity();
    truth.block<3, 3>(0, 0) = Eigen::AngleAxisf(0.10f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    truth.block<3, 1>(0, 3) = Eigen::Vector3f(0.5f, -0.3f, 0.2f);
    auto target = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::transformPointCloud(*source, *target, truth);
    fast_gicp::NDTCuda<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setResolution(1.0);
    ndt.setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
    ndt.setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
    ndt.setTransformationEpsilon(0.001);
    ndt.setRotationEpsilon(0.001);
    ndt.setMaximumIterations(60);
    ndt.setInputSource(source);
    ndt.setInputTarget(target);
    Eigen::Matrix4f guess = truth;
    guess(0, 3) += 0.10f;
    guess(1, 3) -= 0.08f;
    pcl::PointCloud<pcl::PointXYZ> aligned;
    ndt.align(aligned, guess);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    const Eigen::Matrix4f pose = ndt.getFinalTransformation();
    const float translation_error = (pose.block<3, 1>(0, 3) - truth.block<3, 1>(0, 3)).norm();
    const float rotation_error = Eigen::AngleAxisf(truth.block<3, 3>(0, 0).transpose() * pose.block<3, 3>(0, 0)).angle();
    const float initial_error = (guess.block<3, 1>(0, 3) - truth.block<3, 1>(0, 3)).norm();
    std::cout << "converged=" << ndt.hasConverged() << " translation_error_m=" << translation_error
              << " rotation_error_rad=" << rotation_error << std::endl;
    // P2D voxel statistics introduce a sub-voxel bias. Require improvement and
    // a bounded transform error; a second application of the guess fails this.
    return ndt.hasConverged() && pose.allFinite() && translation_error < 0.15f &&
           translation_error < initial_error * 0.9f && rotation_error < 0.03f ? 0 : 1;
}

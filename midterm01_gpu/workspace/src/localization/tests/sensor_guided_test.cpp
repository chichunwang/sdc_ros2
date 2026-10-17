#include "localization/sensor_guided.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <Eigen/Geometry>

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char**) {
    try {
        using Cloud = pcl::PointCloud<pcl::PointXYZ>;
        auto source = std::make_shared<Cloud>();
        auto target = std::make_shared<Cloud>();
        // A nonplanar, deterministic scene in kilometre-scale map coordinates.
        const Eigen::Matrix3f truth_rotation = Eigen::AngleAxisf(0.4f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        const Eigen::Vector3f truth_position(-4800, 2274, 110);
        for (int i = 0; i < 140; ++i) {
            pcl::PointXYZ point;
            point.getVector3fMap() = Eigen::Vector3f(0.7f + 0.09f * i, 2.0f * std::sin(0.41f * i), 0.7f * std::cos(0.31f * i));
            source->push_back(point);
            pcl::PointXYZ map_point;
            map_point.getVector3fMap() = truth_rotation * point.getVector3fMap() + truth_position;
            target->push_back(map_point);
        }
        Eigen::Matrix4f prior = Eigen::Matrix4f::Identity();
        prior.block<3, 3>(0, 0) = truth_rotation;
        prior.block<3, 1>(0, 3) = truth_position + Eigen::Vector3f(0.4f, -0.3f, 0.1f);
        Eigen::Matrix4f bad_seed = prior;
        bad_seed(0, 3) += 100;
        Eigen::Matrix4f output;
        Eigen::Vector4d correction;
        localization::GuidedMetrics metrics;
        Require(localization::RefineSensorGuided(source, target, prior, bad_seed, true,
            Eigen::Vector4d::Zero(), 1.0, 1.5, 8 * M_PI / 180, output, correction, metrics, argc > 1), "Expected valid map update");
        Require(!metrics.used_cuda_seed, "Unbounded CUDA seed must be rejected");
        Require(metrics.final_cost < metrics.prior_cost, "Objective must improve");
        Require(metrics.final_rmse < metrics.prior_rmse, "Known-scene residual must improve");
        Require((output.block<3, 1>(0, 3) - truth_position).norm() < (prior.block<3, 1>(0, 3) - truth_position).norm(), "Known translation must improve");
        Require(correction.head<2>().norm() <= 1.5 && std::abs(correction[3]) <= 8 * M_PI / 180, "Correction bounds violated");
        Require((output.block<3, 3>(0, 0).transpose() * output.block<3, 3>(0, 0) - Eigen::Matrix3f::Identity()).norm() < 1e-5, "Output must remain a rotation");
        // No overlap: explicitly use the current prior, with no stale correction.
        for (auto& point : *target) point.x += 500;
        correction.setConstant(5);
        localization::GuidedMetrics no_overlap;
        Require(!localization::RefineSensorGuided(source, target, prior, bad_seed, true,
            Eigen::Vector4d::Constant(5), .25, 1.5, 8 * M_PI / 180, output, correction, no_overlap, argc > 1), "No overlap must reject scan");
        Require(output.isApprox(prior) && correction.isZero(), "No overlap must use current sensor prior");
        std::cout << "Known transform, large origin, rejected bad CUDA seed, bounded update, and no-overlap prior fallback passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

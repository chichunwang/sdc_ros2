#include "localization/sensor_guided.h"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Cloud::Ptr Plane(bool wall, float origin = 0, int span = 12) {
    auto cloud = std::make_shared<Cloud>();
    for (int i = -span; i <= span; ++i) for (int j = -span; j <= span; ++j) {
        if (wall) cloud->push_back(pcl::PointXYZ(origin, i * .5f, j * .5f));
        else cloud->push_back(pcl::PointXYZ(origin + i * .5f, j * .5f, 0));
    }
    return cloud;
}

int main(int argc, char**) {
    try {
        using namespace localization;
        GeometryConfig config;
        config.mode = GeometryMode::Diagnostic;
        const bool cuda = argc > 1;
        auto target = Plane(false, 4000);
        SurfaceGeometryCache cache(target, config);
        const auto& center = cache.At(312);
        Require(center.reliable && std::abs(center.normal.z()) > .999, "Ground plane normal must be reliable at large origin");
        Require(cache.queries() == 1, "One PCA query expected");
        cache.At(312);
        Require(cache.queries() == 1, "Repeated geometry query must use cache");
        auto line = std::make_shared<Cloud>();
        for (int i = 0; i < 30; ++i) line->push_back(pcl::PointXYZ(i * .2f, 0, 0));
        SurfaceGeometryCache line_cache(line, config);
        Require(!line_cache.At(15).reliable, "A line must not be treated as a reliable plane");
        auto sparse = std::make_shared<Cloud>();
        sparse->push_back(pcl::PointXYZ(0, 0, 0));
        SurfaceGeometryCache sparse_cache(sparse, config);
        Require(!sparse_cache.At(0).reliable, "Insufficient neighbors must reject geometry");

        auto source = std::make_shared<Cloud>();
        for (int i = -7; i <= 7; ++i) for (int j = -7; j <= 7; ++j)
            source->push_back(pcl::PointXYZ(i * .5f, j * .5f, 0));
        Eigen::Matrix4f prior = Eigen::Matrix4f::Identity();
        prior.block<3, 1>(0, 3) = Eigen::Vector3f(4000.15f, -.1f, .3f);
        Eigen::Matrix4f off_output, diagnostic_output, directional_output;
        Eigen::Vector4d off_correction, diagnostic_correction, directional_correction;
        GuidedMetrics off, diagnostic, directional;
        Require(RefineSensorGuided(source, target, prior, prior, false, Eigen::Vector4d::Zero(),
            1., 1.5, 8 * M_PI / 180, off_output, off_correction, off, cuda), "Original point-to-point update failed");
        Require(RefineSensorGuided(source, target, prior, prior, false, Eigen::Vector4d::Zero(),
            1., 1.5, 8 * M_PI / 180, diagnostic_output, diagnostic_correction, diagnostic, cuda, config), "Diagnostic update failed");
        Require((off_output.array() == diagnostic_output.array()).all() &&
            (off_correction.array() == diagnostic_correction.array()).all() && off.final_cost == diagnostic.final_cost,
            "Diagnostic mode must preserve original result and objective exactly");
        Require(diagnostic.geometry.observable_rank == 1, "Ground-only data must expose rank 1, even with full-rank prior");
        Require(diagnostic.geometry.scaled_information(0, 0) < 1e-10 &&
            diagnostic.geometry.scaled_information(1, 1) < 1e-10 && diagnostic.geometry.scaled_information(2, 2) > 0,
            "Ground normals must constrain Z, not fabricate XY information");

        config.mode = GeometryMode::Directional;
        config.tangent_weight = 0.;
        Require(RefineSensorGuided(source, target, prior, prior, false, Eigen::Vector4d::Zero(),
            1., 1.5, 8 * M_PI / 180, directional_output, directional_correction, directional, cuda, config), "Directional ground update failed");
        Require(directional_correction.head<2>().norm() < 1e-8 && std::abs(directional_correction[3]) < 1e-8,
            "Zero tangent weight must leave weak XY/yaw directions at sensor prior");
        Require(std::abs(directional_output(2, 3)) < std::abs(prior(2, 3)) &&
            directional.final_cost < directional.prior_cost && directional.geometry.directional_update_used,
            "Reliable normal direction must improve while preserving weak directions");
        Require(directional.update_status == GuidedUpdateStatus::Accepted &&
            directional.candidate_unique_targets >= 6, "Accepted map update must retain candidate support/status");

        // Many source points on one target are not independent map support.
        auto clustered_source = std::make_shared<Cloud>();
        for (int i = 0; i < 20; ++i) clustered_source->push_back(pcl::PointXYZ(0, 0, 0));
        prior.setIdentity();
        prior(0, 3) = 4000;
        prior(2, 3) = .3f;
        GuidedMetrics clustered;
        Require(!RefineSensorGuided(clustered_source, target, prior, prior, false, Eigen::Vector4d::Zero(),
            .25, 1.5, 8 * M_PI / 180, directional_output, directional_correction, clustered, cuda, config),
            "Repeated matches on one target must not bypass support protection");
        Require(clustered.update_status == GuidedUpdateStatus::InsufficientUniqueTargets &&
            clustered.candidate_reliable_points == 20 && clustered.candidate_unique_targets == 1 &&
            IsGeometrySupportFallback(clustered.update_status), "Must report candidate unique-target rejection explicitly");
        Require(directional_output.isApprox(prior) && directional_correction.isZero(),
            "Unique-target rejection must emit current prior");

        // Reused metrics must not report a stale accepted update on early exit.
        auto tiny_source = std::make_shared<Cloud>();
        tiny_source->push_back(pcl::PointXYZ(0, 0, 0));
        Require(!RefineSensorGuided(tiny_source, target, prior, prior, false, Eigen::Vector4d::Zero(),
            .25, 1.5, 8 * M_PI / 180, directional_output, directional_correction, directional, cuda, config) &&
            directional.update_status == GuidedUpdateStatus::InsufficientInputPoints &&
            !directional.geometry.directional_update_used && directional.candidate_unique_targets == 0,
            "Early exit must reset metrics and record its own reason");

        auto wall = Plane(true, 0, 24);
        auto wall_source = std::make_shared<Cloud>();
        for (int i = -20; i <= 20; ++i) for (int j = -7; j <= 7; ++j)
            wall_source->push_back(pcl::PointXYZ(0, i * .5f, j * .5f));
        prior.setIdentity();
        prior(0, 3) = .3f;
        GuidedMetrics wall_metrics;
        Require(RefineSensorGuided(wall_source, wall, prior, prior, false, Eigen::Vector4d::Zero(),
            1., 1.5, 8 * M_PI / 180, directional_output, directional_correction, wall_metrics, cuda, config), "Wall update failed");
        Require(wall_metrics.geometry.observable_rank == 2, "Wall data must expose X/yaw directions only");
        Require(std::abs(directional_correction[1]) < 1e-8 && std::abs(directional_correction[2]) < 1e-8,
            "Wall tangential Y/Z must remain sensor anchored");

        auto line_source = std::make_shared<Cloud>(*line);
        prior.setIdentity();
        prior(1, 3) = .2f;
        GuidedMetrics invalid;
        Require(!RefineSensorGuided(line_source, line, prior, prior, false, Eigen::Vector4d::Constant(.2),
            .25, 1.5, 8 * M_PI / 180, directional_output, directional_correction, invalid, cuda, config), "Unreliable geometry must reject directional update");
        Require(directional_output.isApprox(prior) && directional_correction.isZero() &&
            !invalid.geometry.directional_update_used, "Rejected geometry must use current prior without stale correction");
        Require(invalid.update_status == GuidedUpdateStatus::InsufficientReliableGeometry &&
            invalid.candidate_reliable_points == 0, "Unreliable surfaces must report geometry support rejection");
        for (auto& point : *wall) point.x += 100;
        GuidedMetrics no_overlap;
        Require(!RefineSensorGuided(wall_source, wall, prior, prior, false, Eigen::Vector4d::Constant(.2),
            .25, 1.5, 8 * M_PI / 180, directional_output, directional_correction, no_overlap, cuda, config), "No-overlap must reject");
        Require(directional_output.isApprox(prior) && directional_correction.isZero(), "No-overlap must not carry previous pose");
        Require(no_overlap.update_status == (cuda ? GuidedUpdateStatus::NoMapCandidates : GuidedUpdateStatus::InsufficientMatches) &&
            !IsGeometrySupportFallback(no_overlap.update_status), "No-overlap must remain distinct from surface quality rejection");
        std::cout << "Geometry PCA/cache, line/sparse rejection, exact diagnostic compatibility, ground/wall degeneracy, directional anchoring, and prior fallback passed ("
                  << (cuda ? "CUDA NN" : "CPU test NN") << ").\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

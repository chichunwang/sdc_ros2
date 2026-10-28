#pragma once

#include "localization/geometry_weighting.h"

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace localization {
enum class GuidedUpdateStatus {
    NotEvaluated, Accepted, InsufficientInputPoints, NoMapCandidates,
    NonfiniteObjective, InsufficientMatches, InsufficientReliableGeometry,
    InsufficientUniqueTargets, ObjectiveIncreased
};
const char* GuidedUpdateStatusName(GuidedUpdateStatus status);
bool IsGeometrySupportFallback(GuidedUpdateStatus status);

struct GuidedMetrics {
    double prior_cost = 0.0;
    double final_cost = 0.0;
    double prior_rmse = 0.0;
    double final_rmse = 0.0;
    int inliers = 0;
    int iterations = 0;
    bool used_cuda_seed = false;
    GuidedUpdateStatus update_status = GuidedUpdateStatus::NotEvaluated;
    // Candidate support BEFORE rejection. geometry below describes the returned
    // pose (the current sensor prior on rejection), which can have other matches.
    int candidate_inliers = 0, candidate_reliable_points = 0, candidate_unique_targets = 0;
    GeometryMetrics geometry;
};

// Fixed IMU roll/pitch, world-frame xyz/yaw correction, robust point-to-map
// likelihood plus GNSS/IMU priors. All matrices map scaled source -> map.
bool RefineSensorGuided(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                        const Eigen::Matrix4f& prior, const Eigen::Matrix4f& cuda_seed,
                        bool cuda_valid, const Eigen::Vector4d& previous_correction,
                        double smoothing_alpha, double max_xy, double max_yaw,
                        Eigen::Matrix4f& output, Eigen::Vector4d& correction,
                        GuidedMetrics& metrics, bool use_cuda_nn = false,
                        const GeometryConfig& geometry_config = GeometryConfig());
}  // namespace localization

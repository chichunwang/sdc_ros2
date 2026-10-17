#pragma once

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace localization {
struct GuidedMetrics {
    double prior_cost = 0.0;
    double final_cost = 0.0;
    double prior_rmse = 0.0;
    double final_rmse = 0.0;
    int inliers = 0;
    int iterations = 0;
    bool used_cuda_seed = false;
};

// Fixed IMU roll/pitch, world-frame xyz/yaw correction, robust point-to-map
// likelihood plus GNSS/IMU priors. All matrices map scaled source -> map.
bool RefineSensorGuided(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                        const Eigen::Matrix4f& prior, const Eigen::Matrix4f& cuda_seed,
                        bool cuda_valid, const Eigen::Vector4d& previous_correction,
                        double smoothing_alpha, double max_xy, double max_yaw,
                        Eigen::Matrix4f& output, Eigen::Vector4d& correction,
                        GuidedMetrics& metrics, bool use_cuda_nn = false);
}  // namespace localization

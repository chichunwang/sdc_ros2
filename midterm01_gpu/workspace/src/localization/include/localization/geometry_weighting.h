#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace localization {
enum class GeometryMode { Off, Diagnostic, Directional };

struct GeometryConfig {
    GeometryMode mode = GeometryMode::Off;
    double normal_gain = 2.0;
    double tangent_weight = 0.05;
    double radius_m = 2.5;
    double min_planarity = 0.2;
    int neighbors = 16;
    int min_neighbors = 6;
    bool enabled() const { return mode != GeometryMode::Off; }
    bool directional() const { return mode == GeometryMode::Directional; }
    void Validate() const;
};

struct SurfaceGeometry {
    bool reliable = false;
    double confidence = 0.0;
    double planarity = 0.0;
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::Matrix3d metric = Eigen::Matrix3d::Zero();
};

// CPU neighborhood PCA only. Registration nearest neighbors remain on CUDA.
class SurfaceGeometryCache {
 public:
    SurfaceGeometryCache(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, const GeometryConfig& config);
    ~SurfaceGeometryCache();
    const SurfaceGeometry& At(int target_index);
    int queries() const;
 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct GeometryMetrics {
    int input_points = 0, euclidean_inliers = 0, reliable_points = 0, unique_targets = 0;
    int pca_queries = 0, observable_rank = 0;
    bool directional_update_used = false;
    double mean_confidence = 0.0, plane_rmse_m = 0.0, condition = 0.0;
    Eigen::Vector3d support_span_m = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_eigenvalues = Eigen::Vector3d::Zero();
    // Data-only normal information, scaled by S=diag(1m,1m,1m,3deg).
    // Eigenvectors refer to scaled coordinates; this is not pose covariance.
    Eigen::Matrix4d scaled_information = Eigen::Matrix4d::Zero();
    Eigen::Vector4d eigenvalues = Eigen::Vector4d::Zero();
    Eigen::Matrix4d eigenvectors = Eigen::Matrix4d::Identity();
};
void FinishGeometryMetrics(GeometryMetrics& metrics, const Eigen::Matrix3d& normal_information);
void WriteGeometryConfigJson(std::ostream& stream, const GeometryConfig& config);
void WriteGeometryCsvHeader(std::ostream& stream);
void WriteGeometryCsvRow(std::ostream& stream, std::size_t idx, std::int64_t timestamp,
                         bool map_accepted, const GeometryMetrics& metrics);
}  // namespace localization

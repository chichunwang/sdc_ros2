#include "localization/geometry_weighting.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>

namespace localization {
void GeometryConfig::Validate() const {
    if (!std::isfinite(normal_gain) || normal_gain <= 0 || normal_gain > 10 ||
        !std::isfinite(tangent_weight) || tangent_weight < 0 || tangent_weight > 1 ||
        !std::isfinite(radius_m) || radius_m <= 0 ||
        !std::isfinite(min_planarity) || min_planarity <= 0 || min_planarity >= 1 ||
        min_neighbors < 6 || neighbors < min_neighbors || (generalized() && tangent_weight <= 0))
        throw std::invalid_argument("Invalid geometry weighting configuration");
}

struct SurfaceGeometryCache::Impl {
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud;
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    GeometryConfig config;
    std::vector<SurfaceGeometry> surfaces;
    std::vector<bool> computed;
    int queries = 0;
    Impl(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input, const GeometryConfig& settings)
        : cloud(input), config(settings), surfaces(input->size()), computed(input->size(), false) {
        config.Validate();
        if (!cloud->empty()) tree.setInputCloud(cloud);
    }
};
SurfaceGeometryCache::SurfaceGeometryCache(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, const GeometryConfig& config)
    : impl_(std::make_unique<Impl>(cloud, config)) {}
SurfaceGeometryCache::~SurfaceGeometryCache() = default;
int SurfaceGeometryCache::queries() const { return impl_->queries; }

const SurfaceGeometry& SurfaceGeometryCache::At(int index) {
    auto& p = *impl_;
    if (index < 0 || static_cast<std::size_t>(index) >= p.cloud->size())
        throw std::out_of_range("Invalid geometry target index");
    if (p.computed[index]) return p.surfaces[index];
    p.computed[index] = true;
    ++p.queries;
    auto& result = p.surfaces[index];
    std::vector<int> neighbors;
    std::vector<float> squared;
    const auto& center = (*p.cloud)[index];
    const int found = p.tree.nearestKSearch(center, p.config.neighbors, neighbors, squared);
    std::vector<Eigen::Vector3d> local;
    for (int i = 0; i < found; ++i) {
        if (squared[i] <= p.config.radius_m * p.config.radius_m)
            local.push_back((*p.cloud)[neighbors[i]].getVector3fMap().cast<double>() - center.getVector3fMap().cast<double>());
    }
    if (local.size() < static_cast<std::size_t>(p.config.min_neighbors)) return result;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& point : local) mean += point;
    mean /= local.size();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& point : local) covariance.noalias() += (point - mean) * (point - mean).transpose();
    covariance /= local.size();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(covariance);
    if (eigen.info() != Eigen::Success) return result;
    const Eigen::Vector3d values = eigen.eigenvalues().cwiseMax(0.0);
    // Reject line-like, tiny and thick/nonplanar neighborhoods. Normal signs do
    // not matter because all residual metrics use n*n^T.
    if (values[1] < 0.04 || values[2] < 1e-8 || values[0] > .25 * values[1]) return result;
    result.planarity = (values[1] - values[0]) / values[2];
    if (result.planarity < p.config.min_planarity) return result;
    result.normal = eigen.eigenvectors().col(0);
    result.confidence = std::clamp(result.planarity, 0.0, 1.0);
    const Eigen::Matrix3d normal_metric = result.normal * result.normal.transpose();
    result.metric = normal_metric + p.config.tangent_weight * (Eigen::Matrix3d::Identity() - normal_metric);
    result.reliable = true;
    return result;
}

void FinishGeometryMetrics(GeometryMetrics& metrics, const Eigen::Matrix3d& normal_information) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen(metrics.scaled_information);
    if (eigen.info() != Eigen::Success) throw std::runtime_error("Geometry information eigensolve failed");
    metrics.eigenvalues = eigen.eigenvalues().cwiseMax(0.0);
    metrics.eigenvectors = eigen.eigenvectors();
    const double largest = metrics.eigenvalues.maxCoeff();
    const double threshold = std::max(1e-4, .05 * largest);
    metrics.observable_rank = (metrics.eigenvalues.array() >= threshold).count();
    metrics.condition = largest > 0 ? largest / std::max(1e-12, metrics.eigenvalues.minCoeff()) : 0.0;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> normals(normal_information);
    if (normals.info() != Eigen::Success) throw std::runtime_error("Normal information eigensolve failed");
    metrics.normal_eigenvalues = normals.eigenvalues().cwiseMax(0.0);
}

void WriteGeometryCsvHeader(std::ostream& s) {
    s << "idx,timestamp_ns,map_accepted,directional_update_used,input_points,euclidean_inliers,reliable_points,unique_targets,pca_queries,mean_confidence,plane_rmse_m,observable_rank,scaled_condition,span_x_m,span_y_m,span_z_m";
    for (int i = 0; i < 3; ++i) s << ",normal_eigenvalue_" << i;
    for (int i = 0; i < 4; ++i) s << ",scaled_eigenvalue_" << i;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) s << ",scaled_eigenvector_r" << r << "c" << c;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) s << ",scaled_information_r" << r << "c" << c;
    s << '\n';
}
void WriteGeometryConfigJson(std::ostream& s, const GeometryConfig& c) {
    const char* mode = c.generalized() ? "gicp" : (c.directional() ? "directional" : (c.enabled() ? "diagnostic" : "off"));
    s << "{\n  \"algorithm_version\": 1,\n  \"mode\": \"" << mode
      << "\",\n  \"normal_gain\": " << c.normal_gain << ",\n  \"tangent_weight\": " << c.tangent_weight
      << ",\n  \"radius_m\": " << c.radius_m << ",\n  \"neighbors\": " << c.neighbors
      << ",\n  \"min_neighbors\": " << c.min_neighbors << ",\n  \"min_planarity\": " << c.min_planarity
      << ",\n  \"min_second_covariance_eigenvalue_m2\": 0.04,\n  \"max_thickness_ratio\": 0.25,"
      << "\n  \"scaled_state_units\": [1, 1, 1, " << 3.0 * std::acos(-1.0) / 180.0 << "],"
      << "\n  \"rank_relative_threshold\": 0.05,\n  \"rank_absolute_threshold\": 0.0001,"
      << "\n  \"min_reliable_points\": 10,\n  \"min_unique_targets\": 6,"
      << "\n  \"information_scope\": \"data-only normal information; excludes prior, tangent weight and normal gain; not covariance\"";
    if (c.generalized()) s << ",\n  \"gicp_model\": \"robust symmetric source/target regularized surface covariance; each contributes half variance; same 4DOF prior and bounds\","
        "\n  \"gicp_covariance_calibrated\": false,\n  \"gicp_missing_source_geometry\": \"constant unmatched cost; no directional fallback\","
        "\n  \"gicp_diagnostic_scope\": \"target-normal information over mutually supported pairs, not full GICP Hessian\"";
    s << "\n}\n";
}
void WriteGeometryCsvRow(std::ostream& s, std::size_t idx, std::int64_t timestamp,
                         bool accepted, const GeometryMetrics& m) {
    s << idx << ',' << timestamp << ',' << accepted << ',' << m.directional_update_used << ','
      << m.input_points << ',' << m.euclidean_inliers << ',' << m.reliable_points << ',' << m.unique_targets << ','
      << m.pca_queries << ',' << m.mean_confidence << ',' << m.plane_rmse_m << ',' << m.observable_rank << ',' << m.condition;
    for (int i = 0; i < 3; ++i) s << ',' << m.support_span_m[i];
    for (int i = 0; i < 3; ++i) s << ',' << m.normal_eigenvalues[i];
    for (int i = 0; i < 4; ++i) s << ',' << m.eigenvalues[i];
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) s << ',' << m.eigenvectors(r, c);
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) s << ',' << m.scaled_information(r, c);
    s << '\n';
}
}  // namespace localization

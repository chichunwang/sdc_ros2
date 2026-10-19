#include "localization/sensor_guided.h"
#include "localization/gicp_metric.h"
#include <cuda_ndt_backend/nearest_neighbor.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <pcl/kdtree/kdtree_flann.h>

namespace localization {
namespace {
using V4 = Eigen::Vector4d;
struct Evaluation {
    double cost = 0.0, mse = 0.0;
    int inliers = 0;
    Eigen::Matrix4d hessian = Eigen::Matrix4d::Zero();
    V4 gradient = V4::Zero();
    int reliable_inliers = 0;
    GeometryMetrics geometry;
};
double Wrap(double angle) { return std::atan2(std::sin(angle), std::cos(angle)); }
}  // namespace

const char* GuidedUpdateStatusName(GuidedUpdateStatus status) {
    switch (status) {
        case GuidedUpdateStatus::NotEvaluated: return "not_evaluated";
        case GuidedUpdateStatus::Accepted: return "accepted";
        case GuidedUpdateStatus::InsufficientInputPoints: return "insufficient_input_points";
        case GuidedUpdateStatus::NoMapCandidates: return "no_map_candidates";
        case GuidedUpdateStatus::NonfiniteObjective: return "nonfinite_objective";
        case GuidedUpdateStatus::InsufficientMatches: return "insufficient_matches";
        case GuidedUpdateStatus::InsufficientReliableGeometry: return "insufficient_reliable_geometry";
        case GuidedUpdateStatus::InsufficientUniqueTargets: return "insufficient_unique_targets";
        case GuidedUpdateStatus::ObjectiveIncreased: return "objective_increased";
    }
    return "unknown";
}
bool IsGeometrySupportFallback(GuidedUpdateStatus status) {
    return status == GuidedUpdateStatus::InsufficientReliableGeometry ||
           status == GuidedUpdateStatus::InsufficientUniqueTargets;
}

bool RefineSensorGuided(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
                        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
                        const Eigen::Matrix4f& prior, const Eigen::Matrix4f& cuda_seed,
                        bool cuda_valid, const Eigen::Vector4d& previous_correction,
                        double smoothing_alpha, double max_xy, double max_yaw,
                        Eigen::Matrix4f& output, Eigen::Vector4d& correction,
                        GuidedMetrics& metrics, bool use_cuda_nn, const GeometryConfig& geometry_config,
                        const HeightConstraint& height_constraint) {
    geometry_config.Validate();
    if (height_constraint.anchored && (!std::isfinite(height_constraint.anchor_z) ||
        std::abs(prior(2, 3) - height_constraint.anchor_z) > HeightConstraint::max_offset + 1e-5))
        throw std::invalid_argument("Anchored height prior must lie inside the absolute altitude bound");
    metrics = GuidedMetrics{};
    const Eigen::Matrix4f saved_seed = cuda_seed;
    correction.setZero();
    output = prior;
    metrics.geometry.input_points = source->size();
    if (source->size() < 10 || target->size() < 10) {
        metrics.update_status = GuidedUpdateStatus::InsufficientInputPoints;
        return false;
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    if (!use_cuda_nn) tree.setInputCloud(target);
    const Eigen::Matrix3d rotation_prior = prior.block<3, 3>(0, 0).cast<double>();
    const Eigen::Vector3d translation_prior = prior.block<3, 1>(0, 3).cast<double>();
    std::unique_ptr<SurfaceGeometryCache> geometry;
    if (geometry_config.enabled()) geometry = std::make_unique<SurfaceGeometryCache>(target, geometry_config);
    std::vector<SurfaceGeometry> source_geometry;
    if (geometry_config.generalized()) {
        SurfaceGeometryCache cache(source, geometry_config);
        source_geometry.reserve(source->size());
        for (std::size_t i = 0; i < source->size(); ++i) {
            source_geometry.push_back(cache.At(static_cast<int>(i)));
            metrics.source_reliable_points += source_geometry.back().reliable;
        }
    }
    std::unique_ptr<cuda_ndt_backend::NearestNeighbor> gpu_nn;
    std::vector<int> gpu_to_target;
    if (use_cuda_nn) {
        Eigen::Vector3d low = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d high = -low;
        double radius = 0.0;
        for (const auto& point : *source) {
            const Eigen::Vector3d rotated = rotation_prior * point.getVector3fMap().cast<double>();
            low = low.cwiseMin(rotated);
            high = high.cwiseMax(rotated);
            radius = std::max(radius, rotated.head<2>().norm());
        }
        // Any neighbor capable of influencing the 2 m capped objective lies
        // within this box for every allowed xyz/yaw correction. The angular
        // arc bound radius*angle covers the entire bounded optimization path.
        const Eigen::Vector3d margin(max_xy + radius * max_yaw + 2.01,
                                      max_xy + radius * max_yaw + 2.01,
                                      (height_constraint.anchored ? 2.0 : 1.0) * HeightConstraint::max_offset + 2.01);
        low -= margin;
        high += margin;
        std::vector<float> local_target;
        local_target.reserve(target->size() * 3);
        for (std::size_t index = 0; index < target->size(); ++index) {
            const auto& point = (*target)[index];
            const Eigen::Vector3f local = (point.getVector3fMap().cast<double>() - translation_prior).cast<float>();
            if ((local.cast<double>().array() < low.array()).any() || (local.cast<double>().array() > high.array()).any()) continue;
            local_target.insert(local_target.end(), {local.x(), local.y(), local.z()});
            gpu_to_target.push_back(static_cast<int>(index));
        }
        if (local_target.empty()) {
            // No map point can enter the inlier region. This is an explicit
            // sensor-only result, not a failed CUDA call or CPU fallback.
            metrics.prior_cost = metrics.final_cost = (0.5 * (2.0 - 0.25)) / (0.3 * 0.3) *
                (geometry_config.directional() ? geometry_config.normal_gain : 1.0);
            if (height_constraint.anchored) {
                const double offset = height_constraint.Offset(translation_prior.z(), 0.0);
                metrics.prior_cost += 0.5 * offset * offset;
                metrics.final_cost = metrics.prior_cost;
            }
            metrics.prior_rmse = metrics.final_rmse = 2.0;
            metrics.update_status = GuidedUpdateStatus::NoMapCandidates;
            return false;
        }
        gpu_nn = std::make_unique<cuda_ndt_backend::NearestNeighbor>(local_target, static_cast<int>(source->size()));
    }
    // Standard deviations: GNSS XY 0.5 m; height regularizer 1 m; IMU yaw 3 deg.
    // Anchored Z penalizes the absolute nominal-altitude offset, not another
    // increment about the previously refined height. GNSS Z may be constant.
    const V4 precision(4.0, 4.0, 1.0, 1.0 / std::pow(3.0 * M_PI / 180.0, 2));
    constexpr double huber = 0.5, cutoff = 2.0, cloud_precision = 1.0 / (0.3 * 0.3);
    const double normalize = cloud_precision / static_cast<double>(source->size());
    const auto valid = [&](const V4& x) {
        return x.allFinite() && x.head<2>().norm() <= max_xy &&
            std::abs(height_constraint.Offset(translation_prior.z(), x[2])) <=
                HeightConstraint::max_offset + (height_constraint.anchored ? 1e-5 : 0.0) &&
            std::abs(x[3]) <= max_yaw;
    };
    const auto evaluate = [&](const V4& x, bool derivatives, bool collect_geometry = false) {
        Evaluation e;
        V4 regularized = x;
        if (height_constraint.anchored) regularized[2] = height_constraint.Offset(translation_prior.z(), x[2]);
        e.cost = 0.5 * regularized.dot(precision.cwiseProduct(regularized));
        e.gradient = precision.cwiseProduct(regularized);
        e.hessian = precision.asDiagonal();
        Eigen::Matrix3d normal_information = Eigen::Matrix3d::Zero();
        Eigen::Vector3d low = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        Eigen::Vector3d high = -low;
        std::vector<bool> seen(collect_geometry ? target->size() : 0, false);
        double confidence_sum = 0.0, plane_squared_sum = 0.0;
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(x[3], Eigen::Vector3d::UnitZ()).toRotationMatrix() * rotation_prior;
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        std::vector<int> gpu_indices;
        if (gpu_nn) {
            std::vector<float> queries;
            queries.reserve(source->size() * 3);
            for (const auto& point : *source) {
                const Eigen::Vector3f local = (rotation * point.getVector3fMap().cast<double>() + x.head<3>()).cast<float>();
                queries.insert(queries.end(), {local.x(), local.y(), local.z()});
            }
            gpu_nn->Query(queries, gpu_indices);
        }
        std::size_t source_index = 0;
        for (const auto& point : *source) {
            const std::size_t current_source_index = source_index++;
            const Eigen::Vector3d rotated = rotation * point.getVector3fMap().cast<double>();
            const Eigen::Vector3d transformed = rotated + translation_prior + x.head<3>();
            pcl::PointXYZ query;
            query.getVector3fMap() = transformed.cast<float>();
            if (gpu_nn) {
                const int local_index = gpu_indices[current_source_index];
                if (local_index < 0 || static_cast<std::size_t>(local_index) >= gpu_to_target.size())
                    throw std::runtime_error("CUDA NN returned an invalid local index");
                indices[0] = gpu_to_target[local_index];
            }
            else if (tree.nearestKSearch(query, 1, indices, distances) != 1) {
                e.cost = std::numeric_limits<double>::infinity();
                return e;
            }
            if (indices[0] < 0 || static_cast<std::size_t>(indices[0]) >= target->size())
                throw std::runtime_error("CUDA NN returned an invalid map index");
            const Eigen::Vector3d residual = transformed - (*target)[indices[0]].getVector3fMap().cast<double>();
            const double length = residual.norm();
            // Constant outlier cost prevents reducing the objective just by losing matches.
            const double clipped = std::min(length, cutoff);
            const double loss = clipped <= huber ? 0.5 * clipped * clipped : huber * (clipped - 0.5 * huber);
            const SurfaceGeometry* surface = nullptr;
            if (geometry && length <= cutoff && (geometry_config.directional() || collect_geometry))
                surface = &geometry->At(indices[0]);
            Eigen::Matrix3d pair_metric, metric_derivative;
            double pair_confidence = surface ? surface->confidence : 0.0;
            if (geometry_config.generalized() && surface && surface->reliable) {
                const auto& source_surface = source_geometry[current_source_index];
                if (!source_surface.reliable) surface = nullptr;
                else {
                    const auto metric = MakeGicpMetric(surface->normal, source_surface.normal, rotation,
                                                       geometry_config.tangent_weight);
                    pair_metric = metric.weight;
                    metric_derivative = metric.yaw_derivative;
                    pair_confidence = std::min(surface->confidence, source_surface.confidence);
                }
            } else if (surface) pair_metric = surface->metric;
            double metric_length = length;
            if (geometry_config.directional()) {
                const double unmatched_loss = huber * (cutoff - .5 * huber);
                double directional_loss = unmatched_loss;
                if (surface && surface->reliable) {
                    metric_length = std::sqrt(std::max(0.0, residual.dot(pair_metric * residual)));
                    const double projected_loss = metric_length <= huber ? .5 * metric_length * metric_length : huber * (metric_length - .5 * huber);
                    // Unreliable/missing matches retain the same constant cost.
                    // Moving to lower confidence cannot lower cost by itself.
                    directional_loss -= pair_confidence * (unmatched_loss - projected_loss);
                    ++e.reliable_inliers;
                }
                e.cost += normalize * geometry_config.normal_gain * directional_loss;
            } else {
                // This path preserves the previously scored objective exactly.
                e.cost += normalize * loss;
            }
            e.mse += clipped * clipped / static_cast<double>(source->size());
            if (length > cutoff) continue;
            ++e.inliers;
            if (!derivatives && !collect_geometry) continue;
            Eigen::Matrix<double, 3, 4> jacobian = Eigen::Matrix<double, 3, 4>::Zero();
            jacobian.block<3, 3>(0, 0).setIdentity();
            jacobian.col(3) = Eigen::Vector3d(-rotated.y(), rotated.x(), 0.0);
            if (collect_geometry && surface && surface->reliable) {
                ++e.geometry.reliable_points;
                confidence_sum += pair_confidence;
                const double plane_residual = surface->normal.dot(residual);
                plane_squared_sum += plane_residual * plane_residual;
                const Eigen::RowVector4d normal_jacobian = surface->normal.transpose() * jacobian;
                const V4 state_scale(1.0, 1.0, 1.0, 3.0 * M_PI / 180.0);
                const Eigen::RowVector4d scaled_jacobian = normal_jacobian * state_scale.asDiagonal();
                const double w = normalize * pair_confidence * std::min(1.0, huber / std::max(std::abs(plane_residual), 1e-12));
                // No prior, tangential floor or normal_gain is included in the
                // diagnostic information; it must reveal planar degeneracy.
                e.geometry.scaled_information.noalias() += w * scaled_jacobian.transpose() * scaled_jacobian;
                normal_information.noalias() += pair_confidence * surface->normal * surface->normal.transpose();
                if (!seen[indices[0]]) {
                    seen[indices[0]] = true;
                    ++e.geometry.unique_targets;
                    const Eigen::Vector3d point = (*target)[indices[0]].getVector3fMap().cast<double>();
                    low = low.cwiseMin(point);
                    high = high.cwiseMax(point);
                }
            }
            if (!derivatives) continue;
            if (geometry_config.directional()) {
                if (!surface || !surface->reliable) continue;
                const double weight = normalize * geometry_config.normal_gain * pair_confidence *
                    std::min(1.0, huber / std::max(metric_length, 1e-12));
                e.hessian.noalias() += weight * jacobian.transpose() * pair_metric * jacobian;
                e.gradient.noalias() += weight * jacobian.transpose() * pair_metric * residual;
                // Source covariance rotates with yaw: retain the derivative so
                // the gradient agrees with the actual line-search objective.
                if (geometry_config.generalized())
                    e.gradient[3] += .5 * weight * residual.dot(metric_derivative * residual);
            } else {
                const double weight = normalize * std::min(1.0, huber / std::max(length, 1e-12));
                e.hessian.noalias() += weight * jacobian.transpose() * jacobian;
                e.gradient.noalias() += weight * jacobian.transpose() * residual;
            }
        }
        if (collect_geometry) {
            e.geometry.input_points = source->size();
            e.geometry.euclidean_inliers = e.inliers;
            e.geometry.pca_queries = geometry->queries();
            if (e.geometry.reliable_points > 0) {
                e.geometry.mean_confidence = confidence_sum / e.geometry.reliable_points;
                e.geometry.plane_rmse_m = std::sqrt(plane_squared_sum / e.geometry.reliable_points);
                normal_information /= e.geometry.reliable_points;
                e.geometry.support_span_m = high - low;
            }
            FinishGeometryMetrics(e.geometry, normal_information);
        }
        return e;
    };
    const Evaluation baseline = evaluate(V4::Zero(), false);
    metrics.prior_cost = baseline.cost;
    metrics.prior_rmse = std::sqrt(baseline.mse);
    V4 x = V4::Zero();
    V4 previous = previous_correction;
    // prior.z already contains the last accepted height. Reusing its old dz
    // as a fresh increment would apply that vertical motion a second time.
    if (height_constraint.anchored) previous[2] = 0.0;
    const V4 carried = previous * 0.8;
    if (valid(carried) && evaluate(carried, false).cost < baseline.cost) x = carried;
    if (cuda_valid) {
        V4 proposal;
        proposal.head<3>() = (saved_seed.block<3, 1>(0, 3) - prior.block<3, 1>(0, 3)).cast<double>();
        proposal[3] = Wrap(std::atan2(saved_seed(1, 0), saved_seed(0, 0)) - std::atan2(prior(1, 0), prior(0, 0)));
        // CUDA roll/pitch are deliberately not propagated into the road-vehicle state.
        if (valid(proposal) && evaluate(proposal, false).cost < evaluate(x, false).cost) {
            x = proposal;
            metrics.used_cuda_seed = true;
        }
    }
    for (int iteration = 0; iteration < 15; ++iteration) {
        const Evaluation current = evaluate(x, true);
        if (current.inliers < 10 || !std::isfinite(current.cost)) break;
        if (geometry_config.directional() && current.reliable_inliers < 10) break;
        V4 step = -current.hessian.ldlt().solve(current.gradient);
        if (!step.allFinite()) break;
        const double scale = std::min({1.0, 0.25 / std::max(step.head<3>().norm(), 1e-12),
                                       (2.0 * M_PI / 180) / std::max(std::abs(step[3]), 1e-12)});
        step *= scale;
        bool accepted = false;
        for (int search = 0; search < 8; ++search) {
            const V4 next = x + step;
            if (valid(next) && evaluate(next, false).cost < current.cost) {
                x = next;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        metrics.iterations = iteration + 1;
        if (!accepted || (step.head<3>().norm() < 1e-3 && std::abs(step[3]) < 1e-4)) break;
    }
    V4 smoothed = (1 - smoothing_alpha) * previous + smoothing_alpha * x;
    // An old correction must never bypass the current sensor bound or objective.
    Evaluation final = valid(smoothed) ? evaluate(smoothed, false) : baseline;
    if (!valid(smoothed) || final.cost > baseline.cost) {
        smoothed = x;
        final = evaluate(x, false);
    }
    if (geometry_config.enabled()) {
        final = evaluate(smoothed, false, true);
        metrics.geometry = final.geometry;
    }
    metrics.candidate_inliers = final.inliers;
    metrics.candidate_reliable_points = metrics.geometry.reliable_points;
    metrics.candidate_unique_targets = metrics.geometry.unique_targets;
    if (!std::isfinite(final.cost)) metrics.update_status = GuidedUpdateStatus::NonfiniteObjective;
    else if (final.inliers < 10) metrics.update_status = GuidedUpdateStatus::InsufficientMatches;
    else if (final.cost > baseline.cost + 1e-9) metrics.update_status = GuidedUpdateStatus::ObjectiveIncreased;
    else if (geometry_config.directional() && metrics.geometry.reliable_points < 10)
        metrics.update_status = GuidedUpdateStatus::InsufficientReliableGeometry;
    else if (geometry_config.directional() && metrics.geometry.unique_targets < 6)
        metrics.update_status = GuidedUpdateStatus::InsufficientUniqueTargets;
    else metrics.update_status = GuidedUpdateStatus::Accepted;
    if (metrics.update_status != GuidedUpdateStatus::Accepted) {
        metrics.final_cost = baseline.cost;
        metrics.final_rmse = metrics.prior_rmse;
        metrics.inliers = baseline.inliers;
        if (geometry_config.enabled()) metrics.geometry = evaluate(V4::Zero(), false, true).geometry;
        return false;
    }
    correction = smoothed;
    output.block<3, 3>(0, 0) = (Eigen::AngleAxisd(correction[3], Eigen::Vector3d::UnitZ()).toRotationMatrix() * rotation_prior).cast<float>();
    output.block<3, 1>(0, 3) = (translation_prior + correction.head<3>()).cast<float>();
    metrics.final_cost = final.cost;
    metrics.final_rmse = std::sqrt(final.mse);
    metrics.inliers = final.inliers;
    metrics.geometry.directional_update_used = geometry_config.directional();
    return true;
}
}  // namespace localization
